"""Bounded real pretrained EfficientNet pilot; no accuracy experiment."""
import inspect
import os
from pathlib import Path
import tempfile
import time
os.environ.setdefault('KERAS_HOME', str(Path('artifacts/method/pilot_keras_cache').resolve()))
import numpy as np
import tensorflow as tf
import keras
from gesture_cv.dataset import sha256, write_json
from gesture_cv.preprocessing import open_rgb, model_input

def main():
    tf.keras.utils.set_random_seed(123)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    paths = [sorted(Path('data/clean/a1_clean_v1', c).glob('*.png'))[0] for c in ['peace','stop','thumbs_up']]
    x = np.concatenate([model_input(open_rgb(p), already_cropped=True) for p in paths])
    report = {'status':'RUNNING','seed':123,'purpose':'Feasibility only; no accuracy or final split use',
              'tensorflow':tf.__version__,'keras':keras.__version__, 'inputs':{str(p):sha256(p) for p in paths},
              'weights_url':'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
              'weights_expected_md5':'50bc09e76180e00e4465e1a485ddc09d',
              'preprocessing':'RGB 224x224 float32 [0,255]; crop already applied; model embeds scaling'}
    output='artifacts/method/pilot_efficientnet.json'
    write_json(output, report)
    try:
        start=time.perf_counter()
        base=tf.keras.applications.EfficientNetB0(include_top=False,weights='imagenet',pooling='avg',input_shape=(224,224,3))
        report['load_seconds']=time.perf_counter()-start
        start=time.perf_counter();features=base(x,training=False).numpy();report['first_batch_seconds']=time.perf_counter()-start
        assert features.shape==(3,1280) and np.isfinite(features).all()
        assert not np.allclose(features[0],features[1])
        times=[]
        for _ in range(3):
            start=time.perf_counter();base(x[:1],training=False).numpy();times.append(time.perf_counter()-start)
        report['warm_single_image_seconds']=times
        base.trainable=False
        head=tf.keras.Sequential([tf.keras.layers.Input((1280,)),tf.keras.layers.Dense(3,activation='softmax')])
        head.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
        loss=head.train_on_batch(features,np.array([0,1,2]));assert np.isfinite(loss).all()
        model=tf.keras.Sequential([tf.keras.layers.Input((224,224,3)),base,head])
        expected=model(x,training=False).numpy()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'pilot.keras';model.save(path)
            reloaded=tf.keras.models.load_model(path,compile=False)
            np.testing.assert_allclose(expected,reloaded(x,training=False).numpy(),atol=1e-5,rtol=1e-4)
        weights=Path(os.environ['KERAS_HOME'])/'models/efficientnetb0_notop.h5'
        report.update(status='PASS',feature_shape=list(features.shape),parameter_count=base.count_params(),
                      weights_sha256=sha256(weights),weights_bytes=weights.stat().st_size,
                      implementation_sha256=sha256(inspect.getfile(tf.keras.applications.EfficientNetB0)),
                      features_differ=True,head_training_finite=True,save_reload_passed=True,
                      device=[d.name for d in tf.config.list_physical_devices()])
    except Exception as e:
        report.update(status='FAIL',error=repr(e));raise
    finally:write_json(output, report)
if __name__=='__main__':main()
