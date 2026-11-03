"""Run after resolving runtime dependencies. This is NOT an accuracy experiment."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from gesture_cv.dataset import write_json, sha256
from gesture_cv import CLASS_NAMES
from gesture_cv.training import build_model
from gesture_cv.preprocessing import probabilities

def main():
    import tensorflow as tf
    import cv2
    report={'tensorflow':tf.__version__,'opencv':cv2.__version__,'numpy':np.__version__,'models':{},'purpose':'runtime compatibility smoke only','class_names':list(CLASS_NAMES),'class_order_evidence':'artifacts/audit/a1_context.json'}
    for path in sorted(Path('models').glob('*.keras')):
        try:
            model=tf.keras.models.load_model(path,compile=False)
            p=model(np.zeros((1,224,224,3),np.float32),training=False).numpy()
            probabilities(p);report['models'][path.name]={'passed':True,'shape':list(p.shape),'sha256':sha256(path)}
        except Exception as exc:report['models'][path.name]={'passed':False,'error':str(exc)}
    with tempfile.TemporaryDirectory() as td:
        model=tf.keras.Sequential([tf.keras.layers.Input((4,)),tf.keras.layers.Dense(3,activation='softmax')])
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
        x=np.zeros((2,4),np.float32);model.train_on_batch(x,np.array([0,1]))
        model.save(Path(td)/'smoke.keras');reloaded=tf.keras.models.load_model(Path(td)/'smoke.keras')
        np.testing.assert_allclose(model(x).numpy(),reloaded(x).numpy(),atol=1e-5,rtol=1e-4)
        report['tiny_training_reload_passed']=True
    # Disposable real classifier architecture, random inputs; not a dataset run.
    classifier,_=build_model(tf,'baseline',123)
    classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    start=time.perf_counter()
    loss=classifier.train_on_batch(np.zeros((1,224,224,3),np.float32),np.array([0]))
    elapsed=time.perf_counter()-start
    assert np.isfinite(loss).all()
    report['baseline_training_smoke']={'batch_size':1,'steps':1,'seconds_including_trace':elapsed,
                                      'finite_loss':True,'parameters':classifier.count_params(),
                                      'purpose':'Compute feasibility only; no accuracy or full-run cost estimate'}
    report['devices']=[d.name for d in tf.config.list_physical_devices()]
    write_json('artifacts/qa/runtime_smoke.json',report)
    if not all(r['passed'] for r in report['models'].values()):raise SystemExit(1)
if __name__=='__main__':main()
