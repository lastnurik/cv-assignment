"""Freeze validation selection, evaluate held-out records once, record real evidence."""
import json,time,gc,base64
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score,f1_score
from gesture_cv import CLASS_NAMES
from gesture_cv.dataset import sha256,write_json
from gesture_cv.preprocessing import open_rgb,model_input,probabilities,DEFAULT
from gesture_cv.metrics import compare_aligned
from gesture_cv.inference import KerasEngine,overlay_heatmap

OUT=Path('artifacts/evaluation');ROLES=['B0','B1','S1','A0']

def main():
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    if (OUT/'selection.json').exists():raise FileExistsError('Evaluation already started; inspect checkpoint before retry')
    manifest_path=Path('data/manifests/a1_limited_v2.json');manifest=json.loads(manifest_path.read_text())
    scores={};reviews={}
    for role in ROLES:
        run=Path('runs')/(role+'_limited_v1');cfg=json.loads((run/'config.json').read_text())
        assert cfg['manifest_sha256']==sha256(manifest_path)
        assert cfg['contract_sha256']==sha256(cfg['contract'])
        assert json.loads((run/'state.json').read_text())['status']=='TRAINED_NOT_TEST_EVALUATED'
        histories={f.name.removesuffix('_history.json'):json.loads(f.read_text()) for f in run.glob('*_history.json')}
        for phase,h in histories.items():
            assert len(h['loss'])==(cfg['fine_tune_epochs'] if phase=='finetune' else cfg['epochs'])
            assert all(np.isfinite(v).all() for v in h.values())
        phase=max(histories,key=lambda p:max(histories[p]['val_accuracy']));h=histories[phase]
        scores[role]=max(h['val_accuracy']);reviews[role]={'selected_phase':phase,'selected_epoch':int(np.argmax(h['val_accuracy']))+1,'validation_accuracy':scores[role],'weights_sha256':sha256(run/'best.keras'),'config_sha256':sha256(run/'config.json')}
    # Verify matched pretraining ablation before test access.
    initial=[]
    for role in ['S1','A0']:
        model=tf.keras.models.load_model(f'runs/{role}_limited_v1/initial.keras',compile=False)
        initial.append(model.layers[-1].get_weights());del model;tf.keras.backend.clear_session();gc.collect()
    for a,b in zip(*initial):np.testing.assert_array_equal(a,b)
    configs=[json.loads(Path(f'runs/{r}_limited_v1/config.json').read_text()) for r in ['S1','A0']]
    differences={k for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
    assert differences=={'run_role','run_id','backbone_weights'},differences
    selected=max(['B0','B1','S1'],key=lambda r:scores[r])
    write_json(OUT/'selection.json',{'role':selected,'rule':'Maximum validation accuracy; first B0/B1/S1 on ties; test unopened','runs':reviews,'manifest_sha256':sha256(manifest_path),'ablation_initial_head_equal':True,'ablation_config_differences':sorted(differences)})
    # First and only final test pass; no selection changes after this point.
    rows=[r for r in manifest['rows'] if r['split']=='test' and r['review_status']=='accepted']
    x=np.concatenate([model_input(open_rgb(r['path']),already_cropped=True) for r in rows]);predictions={}
    for role in ROLES:
        run=Path('runs')/(role+'_limited_v1');model=tf.keras.models.load_model(run/'best.keras',compile=False)
        p=np.concatenate([model(x[i:i+8],training=False).numpy() for i in range(0,len(x),8)])
        result=[]
        for row,values in zip(rows,p):
            values=probabilities(values)
            result.append({'sample_id':row['sample_id'],'label':row['label'],'prediction':CLASS_NAMES[int(np.argmax(values))],'environment':row['environment'],'probabilities':values.tolist(),'group_id':row['group_id']})
        write_json(run/'predictions.json',result);predictions[role]=result
        del model;tf.keras.backend.clear_session();gc.collect()
    comparison=compare_aligned(predictions)
    for role,result in predictions.items():
        truth=[r['label'] for r in result];pred=[r['prediction'] for r in result];metrics=comparison[role]['overall']
        assert abs(metrics['accuracy']-accuracy_score(truth,pred))<1e-6
        assert abs(metrics['macro_f1']-f1_score(truth,pred,labels=list(CLASS_NAMES),average='macro',zero_division=0))<1e-6
        assert sum(map(sum,metrics['confusion_matrix']))==51
    write_json(OUT/'comparison.json',comparison)
    run=Path('runs')/(selected+'_limited_v1')
    # Explicit candidate engine; no public or verified-release claim during checks.
    engine=KerasEngine('.');engine.path=run/'best.keras';engine.label=f'A2 {selected}: limited within-setting evaluation';engine.config=DEFAULT;engine.load()
    sample=x[:1]
    for _ in range(10):engine.model(sample,training=False).numpy()
    times=[]
    for _ in range(100):
        start=time.perf_counter();engine.model(sample,training=False).numpy();times.append((time.perf_counter()-start)*1000)
    write_json(OUT/'latency.json',{'model':selected,'weights_sha256':sha256(run/'best.keras'),'device':engine.status()['device'],'backend':engine.status()['backend'],'tensorflow':tf.__version__,'warmup_count':10,'samples':100,'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'values_ms':times,'cold_warmup_ms':engine.warmup_ms})
    chosen=[]
    for label in CLASS_NAMES:chosen.append(next(i for i,r in enumerate(rows) if r['label']==label))
    errors=[i for i,r in enumerate(predictions[selected]) if r['label']!=r['prediction']]
    if errors and errors[0] not in chosen:chosen.append(errors[0])
    explanations=[]
    for i in chosen:
        row=rows[i];raw=open_rgb(row['source_path']);result=engine.predict(raw)
        np.testing.assert_allclose(list(result['probabilities'].values()),predictions[selected][i]['probabilities'],atol=1e-5,rtol=1e-4)
        target=CLASS_NAMES.index(result['prediction']);heat,nonzero=engine.explain(x[i:i+1],target)
        assert np.isfinite(heat).all()
        png=OUT/(row['sample_id']+'_gradcam.png');png.write_bytes(base64.b64decode(result['heatmap'].split(',',1)[1]))
        other,_=engine.explain(x[i:i+1],(target+1)%3)
        explanations.append({'sample_id':row['sample_id'],'label':row['label'],'prediction':result['prediction'],'file':str(png),'sha256':sha256(png),'nonzero':nonzero,'target_maps_differ':not np.allclose(heat,other),'explanation_ms':result['explanation_ms'],'raw_vs_clean_parity':True})
    assert any(r['nonzero'] and r['target_maps_differ'] for r in explanations)
    write_json(OUT/'explanations.json',explanations)
    write_json('deliverables/model_bundle.json',{'name':f'A2 {selected} — limited within-setting classifier','status':'verified','weights':f'../runs/{selected}_limited_v1/best.keras','weights_sha256':sha256(run/'best.keras'),'class_names':list(CLASS_NAMES),'preprocessing':DEFAULT.to_dict(),'method':'ConvNeXt comparison with pretraining ablation; deployed model selected on validation','scope':'One participant, one indoor setting; not generalization evidence','synthetic_share':0})
    write_json(OUT/'verification.json',{'passed':True,'selected':selected,'aligned_test_samples':51,'sklearn_crosscheck':True,'raw_clean_model_parity':True,'frozen_selection_before_test':True,'ablation_controls_verified':True})
    print('Evaluation verified:',selected)
if __name__=='__main__':main()
