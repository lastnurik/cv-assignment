"""Real Keras inference/Grad-CAM backend. No fallback or fabricated predictions.
Runtime verification remains pending until TensorFlow is installed.
"""
import base64
import io
import json
import platform
import time
from pathlib import Path
import numpy as np
from PIL import Image
from . import CLASS_NAMES
from .dataset import sha256
from .preprocessing import PreprocessConfig, model_input, crop_rgb, probabilities


def image_uri(im):
    buffer=io.BytesIO();im.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def overlay_heatmap(image,heatmap):
    heat=np.asarray(heatmap,dtype=np.float32)
    if heat.ndim!=2 or not np.isfinite(heat).all():raise ValueError('Invalid explanation map')
    values=np.asarray(Image.fromarray(heat).resize(image.size,Image.Resampling.BILINEAR))
    values=np.clip(values,0,1)
    # Transparent amber emphasis, not a random/color-only explanation.
    color=np.zeros((*values.shape,3),dtype=np.float32);color[:]=[255,85,20]
    alpha=(.55*values)[...,None]
    result=np.asarray(image,dtype=np.float32)*(1-alpha)+color*alpha
    return Image.fromarray(np.uint8(np.clip(result,0,255)))


class KerasEngine:
    def __init__(self,root,bundle=None,allow_legacy=False):
        self.root=Path(root).resolve();self.loaded=False;self.error=None
        self.timings=[];self.bundle_path=Path(bundle) if bundle else None
        self.legacy=bool(allow_legacy)
        self.label='No verified A2 model configured'
        self.config=PreprocessConfig()
        if self.bundle_path:
            data=json.loads(self.bundle_path.read_text())
            if data['class_names']!=list(CLASS_NAMES):raise ValueError('Bundle class order mismatch')
            self.path=(self.bundle_path.parent/data['weights']).resolve()
            if sha256(self.path)!=data['weights_sha256']:raise ValueError('Bundle weights hash mismatch')
            self.config=PreprocessConfig(**data['preprocessing']);self.label=data['name']
            if data.get('status')!='verified':raise ValueError('Bundle must reference a verified checkpoint')
        elif self.legacy:
            self.path=self.root/'models/mobilenetv2_finetuned.keras'
            self.config=PreprocessConfig(crop_top_fraction=0,version='legacy_rgb_full_frame')
            self.label='Historical A1 MobileNetV2 — leakage-affected diagnostic only'
        else:
            self.path=None
            self.error='Final A2 model is not trained. Supply a verified model bundle; legacy diagnostics require --allow-legacy.'

    def load(self):
        if self.loaded:return
        if self.path is None:raise RuntimeError(self.error)
        try:
            import tensorflow as tf
            self.tf=tf
            self.model=tf.keras.models.load_model(self.path,compile=False)
            self.layers=[l for l in self.model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
            if not isinstance(self.model,tf.keras.Sequential):
                raise ValueError('This diagnostic backend supports sequential classifier bundles only; verify/export an adapter for another architecture.')
            # Last top-level spatial layer, or nested MobileNet output, before pooling.
            candidates=[]
            for i,l in enumerate(self.layers):
                if isinstance(l,tf.keras.layers.Conv2D):candidates.append(i)
                elif isinstance(l,tf.keras.Model) and l.name!='augmentation':
                    if len(l.output_shape)==4:candidates.append(i)
            if not candidates:raise ValueError('No supported spatial feature layer for Grad-CAM')
            self.feature_index=candidates[-1]
            dummy=np.zeros((1,self.config.height,self.config.width,3),np.float32)
            start=time.perf_counter();probabilities(self.model(dummy,training=False).numpy())
            self.warmup_ms=(time.perf_counter()-start)*1000
            self.loaded=True;self.error=None
        except Exception as exc:
            self.error=f'Model runtime unavailable: {type(exc).__name__}: {exc}'
            raise RuntimeError(self.error) from exc

    def status(self):
        devices=[]
        if self.loaded:devices=[d.name for d in self.tf.config.list_physical_devices()]
        return {'loaded':self.loaded,'model':self.label,'legacy':self.legacy,
                'device':devices or ['not initialized'],'backend':platform.platform(),
                'error':self.error,'synthetic_media_count':0,
                'mean_inference_ms':float(np.mean(self.timings)) if self.timings else None,
                'timed_requests':len(self.timings),'crop_fraction':self.config.crop_top_fraction}

    def explain(self,x,target):
        tf=self.tf
        with tf.GradientTape() as tape:
            z=tf.convert_to_tensor(x);features=None
            for index,layer in enumerate(self.layers):
                if index==len(self.layers)-1:
                    if not isinstance(layer,tf.keras.layers.Dense):raise ValueError('Expected dense output head')
                    # Use logits without mutating the model's softmax activation.
                    logits=tf.linalg.matmul(z,layer.kernel)
                    if layer.use_bias:logits=logits+layer.bias
                    score=logits[:,target]
                else:
                    z=layer(z,training=False)
                    if index==self.feature_index:
                        features=z;tape.watch(features)
        gradients=tape.gradient(score,features)
        if gradients is None:raise RuntimeError('No gradient path; explanation unavailable')
        if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):raise RuntimeError('Non-finite explanation gradients')
        weights=tf.reduce_mean(gradients,axis=(0,1,2))
        heat=tf.nn.relu(tf.reduce_sum(features[0]*weights,axis=-1)).numpy()
        peak=float(heat.max())
        return heat/peak if peak>0 else np.zeros_like(heat),peak>0

    def predict(self,image):
        self.load()
        x=model_input(image,config=self.config)
        start=time.perf_counter();p=probabilities(self.model(x,training=False).numpy())
        elapsed=(time.perf_counter()-start)*1000
        self.timings=(self.timings+[elapsed])[-100:]
        target=int(np.argmax(p));start=time.perf_counter()
        heat,nonzero=self.explain(x,target)
        explanation_ms=(time.perf_counter()-start)*1000
        cropped,box=crop_rgb(image,self.config)
        return {'prediction':CLASS_NAMES[target],'probabilities':dict(zip(CLASS_NAMES,map(float,p))),
                'inference_ms':elapsed,'mean_inference_ms':float(np.mean(self.timings)),
                'timed_requests':len(self.timings),'explanation_ms':explanation_ms,
                'explanation_method':'Grad-CAM (target-class logit)',
                'explanation_valid':nonzero,'explanation_note':'All-zero activation: no positive evidence map.' if not nonzero else 'Diagnostic visualization; not proof of causal reasoning.',
                'image':image_uri(cropped),'heatmap':image_uri(overlay_heatmap(cropped,heat)),
                'crop_box':box,'model':self.label,'legacy':self.legacy,'device':self.status()['device']}
