"""Production constructor/ablation smoke; no final dataset training."""
import os
from pathlib import Path
os.environ.setdefault('KERAS_HOME',str(Path('artifacts/method/pilot_keras_cache').resolve()))
import tempfile
import numpy as np
import tensorflow as tf
from gesture_cv.training import build_model
from gesture_cv.dataset import write_json

def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    heads=[];reports=[]
    for weights in ('imagenet',None):
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(123)
        model,base=build_model(tf,'convnext_tiny',123,weights)
        heads.append([v.copy() for v in model.layers[-1].get_weights()])
        before=base.weights[0].numpy().copy()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
        x=np.zeros((1,224,224,3),np.float32)
        loss=model.train_on_batch(x,np.array([0]));assert np.isfinite(loss).all()
        np.testing.assert_array_equal(before,base.weights[0].numpy())
        expected=model(x,training=False).numpy()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'model.keras';model.save(p)
            reloaded=tf.keras.models.load_model(p,compile=False)
            np.testing.assert_allclose(expected,reloaded(x,training=False).numpy(),atol=1e-5,rtol=1e-4)
        reports.append({'backbone_weights':weights,'finite_loss':True,'frozen_backbone':True,'reload_parity':True})
    for a,b in zip(*heads):np.testing.assert_array_equal(a,b)
    write_json('artifacts/qa/convnext_branch_smoke.json',{'passed':True,'identical_initial_heads':True,'runs':reports,'scope':'Disposable synthetic-input constructor smoke only'})
if __name__=='__main__':main()
