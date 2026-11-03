"""Disposable production-layout/Grad-CAM check for the ConvNeXt candidate."""
import os
from pathlib import Path
os.environ.setdefault('KERAS_HOME',str(Path('artifacts/method/pilot_keras_cache').resolve()))
import tempfile
import numpy as np
import tensorflow as tf
from gesture_cv.dataset import sha256,write_json
from gesture_cv.preprocessing import model_input,open_rgb
from gesture_cv.inference import KerasEngine

def main():
    tf.keras.utils.set_random_seed(123)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    report={'status':'RUNNING','purpose':'Disposable candidate architecture compatibility; not final model','seed':123}
    path='artifacts/method/pilot_convnext_spatial.json';write_json(path,report)
    try:
        paths=[sorted(Path('data/clean/a1_clean_v1',c).glob('*.png'))[0] for c in ['peace','stop','thumbs_up']]
        x=np.concatenate([model_input(open_rgb(p),already_cropped=True) for p in paths])
        report['inputs']={str(p):sha256(p) for p in paths}
        base=tf.keras.applications.ConvNeXtTiny(include_top=False,weights='imagenet',pooling=None,input_shape=(224,224,3))
        base.trainable=False
        pool=tf.keras.layers.GlobalAveragePooling2D();head=tf.keras.layers.Dense(3,activation='softmax')
        features=pool(base(x,training=False)).numpy()
        trainer=tf.keras.Sequential([tf.keras.layers.Input((768,)),head]);trainer.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
        assert np.isfinite(trainer.train_on_batch(features,np.array([0,1,2]))).all()
        model=tf.keras.Sequential([tf.keras.layers.Input((224,224,3)),base,pool,head])
        expected=model(x,training=False).numpy()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pilot.keras';model.save(p)
            loaded=tf.keras.models.load_model(p,compile=False)
            np.testing.assert_allclose(expected,loaded(x,training=False).numpy(),atol=1e-5,rtol=1e-4)
        e=KerasEngine('.');e.tf=tf;e.model=model;e.layers=model.layers;e.feature_index=0
        maps=[];nonzero=[]
        for i in range(3):
            targets=[e.explain(x[i:i+1],j) for j in range(3)]
            assert all(h.shape==(7,7) and np.isfinite(h).all() for h,_ in targets)
            assert any(valid for _,valid in targets)
            assert any(not np.allclose(targets[0][0],h) for h,_ in targets[1:])
            maps.append(targets[0][0]);nonzero.append([v for _,v in targets])
        assert any(not np.allclose(maps[0],h) for h in maps[1:])
        report.update(status='PASS',spatial_shape=list(base.output_shape),save_reload_passed=True,
                      head_training_finite=True,gradcam_finite=True,gradcam_target_input_dependence=True,
                      gradcam_nonzero_by_image_target=nonzero,
                      weights_sha256=sha256(Path(os.environ['KERAS_HOME'])/'models/convnext_tiny_notop.h5'))
    except Exception as ex:report.update(status='FAIL',error=repr(ex));raise
    finally:write_json(path,report)
if __name__=='__main__':main()
