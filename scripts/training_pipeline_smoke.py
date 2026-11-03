"""Disposable artificial fixtures exercise training I/O; never experiment evidence."""
import os
from pathlib import Path
os.environ.setdefault('KERAS_HOME',str(Path('artifacts/method/pilot_keras_cache').resolve()))
import json,tempfile
import numpy as np
from PIL import Image
from gesture_cv.dataset import write_json,sha256
from gesture_cv.preprocessing import DEFAULT
from gesture_cv.training import train
from gesture_cv import CLASS_NAMES

def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);rows=[]
        for split in ['train','val','test']:
            for i,label in enumerate(CLASS_NAMES):
                p=root/f'{split}_{label}.png';Image.new('RGB',(224,224),(40+i*60,30+len(split),70)).save(p)
                rows.append(dict(sample_id=p.stem,path=p.name,sha256=sha256(p),source_sha256=sha256(p),label=label,origin='real',review_status='accepted',consent_status='not_applicable',split=split,group_id=p.stem,environment='ARTIFICIAL_FIXTURE_NOT_REAL_DATA',transform=DEFAULT.to_dict(),already_cropped=True))
        write_json(root/'manifest.json',{'class_names':list(CLASS_NAMES),'rows':rows})
        settings=dict(architecture='convnext_tiny',backbone_weights=None,seed=123,epochs=1,fine_tune_epochs=0,batch_size=3,steps_per_epoch=1,learning_rate=.001,fine_tune_learning_rate=.00001,manifest_sha256=sha256(root/'manifest.json'))
        write_json(root/'contract.json',{'status':'FROZEN','method':'convnext_pretraining','runs':{'A0':settings},'scope':'DISPOSABLE ARTIFICIAL FIXTURE'})
        write_json(root/'config.json',dict(settings,contract='contract.json',contract_sha256=sha256(root/'contract.json'),manifest='manifest.json',run_role='A0',run_id='fixture'))
        run=train(root/'config.json',root)
        ids=json.loads((run/'sample_ids.json').read_text());assert all(not x.startswith('test') for xs in ids.values() for x in xs)
        import tensorflow as tf
        model=tf.keras.models.load_model(run/'best.keras',compile=False)
        features=np.load(run/'features_train_val.npz')['val']
        vx=np.concatenate([np.array(Image.open(root/r['path']),dtype=np.float32)[None] for r in rows if r['split']=='val'])
        expected=model.layers[-1](features).numpy()
        np.testing.assert_allclose(model(vx,training=False).numpy(),expected,atol=1e-5,rtol=1e-4)
        assert (run/'frozen_last.keras').exists()
        write_json('artifacts/qa/training_pipeline_smoke.json',{'passed':True,'scope':'Artificial disposable fixture only','test_excluded':True,'cached_full_model_parity':True,'full_checkpoint_reloaded':True,'history_epochs':len(json.loads((run/'frozen_history.json').read_text())['loss'])})
if __name__=='__main__':main()
