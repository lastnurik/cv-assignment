"""A2 training entry point. Requires reviewed splits and a frozen contract.
Final training requires a frozen contract; architecture pilots are recorded separately.
"""
import json
from pathlib import Path
import time
import numpy as np
from . import CLASS_NAMES
from .dataset import sha256,validate_manifest,write_json
from .preprocessing import open_rgb,model_input,DEFAULT


def preflight(config_path,root):
    root=Path(root).resolve();config=json.loads(Path(config_path).read_text())
    contract_path=root/config['contract'];contract=json.loads(contract_path.read_text())
    if contract.get('status')!='FROZEN':raise ValueError('Experiment contract is not frozen; complete G03/G04/G05 first.')
    if sha256(contract_path)!=config['contract_sha256']:raise ValueError('Contract changed after run configuration')
    approved=contract.get('runs',{}).get(config.get('run_role'))
    if not approved or any(config.get(key)!=value for key,value in approved.items()):
        raise ValueError('Run settings do not match the frozen contract')
    required={'architecture','seed','epochs','fine_tune_epochs','batch_size','steps_per_epoch','learning_rate','fine_tune_learning_rate','manifest_sha256'}
    if not required.issubset(approved):raise ValueError('Frozen contract lacks required controlled settings')
    manifest_path=root/config['manifest'];manifest=json.loads(manifest_path.read_text())
    if sha256(manifest_path)!=config['manifest_sha256']:raise ValueError('Manifest hash mismatch')
    check=validate_manifest(manifest,root)
    if not check['passed']:raise ValueError('Dataset not training-ready: '+'; '.join(check['errors'][:6]))
    accepted=[r for r in manifest['rows'] if r['review_status']=='accepted']
    validate_run_branch(config,contract,accepted)
    for row in accepted:
        if row.get('transform') != DEFAULT.to_dict():raise ValueError('Preprocessing differs from implemented contract')
    for field in ('epochs','batch_size','steps_per_epoch'):
        if not isinstance(config[field],int) or config[field]<=0:raise ValueError(f'Invalid {field}')
    if config['architecture']=='mobilenetv2' and config.get('fine_tune_epochs',0)<1:raise ValueError('Declare positive fine_tune_epochs')
    run=(root/'runs'/config['run_id']).resolve()
    if run.parent != root/'runs' or run.exists():raise ValueError('Run ID unsafe or already exists; never overwrite a run')
    return config,manifest,run


def validate_run_branch(config,contract,accepted):
    role=config['run_role'];architecture=config['architecture']
    synthetic=any(r['origin']=='synthetic' for r in accepted)
    if contract.get('method')=='convnext_pretraining':
        expected={'B0':'baseline','B1':'mobilenetv2','S1':'convnext_tiny','A0':'convnext_tiny'}
        if role not in expected or architecture!=expected[role]:raise ValueError('Invalid ConvNeXt branch role/architecture')
        if synthetic:raise ValueError('ConvNeXt comparison must be real-only')
        if role in ('S1','A0'):
            if config.get('backbone_weights')!=('imagenet' if role=='S1' else None):raise ValueError('Ablation backbone initialization mismatch')
            if config.get('fine_tune_epochs')!=0:raise ValueError('Representation ablation requires frozen backbone')
    else:
        expected={'B0':'baseline','B1':'mobilenetv2','S1':'mobilenetv2'}
        if role not in expected or architecture!=expected[role]:raise ValueError('Invalid synthesis branch role/architecture')
        if role in ('B0','B1') and synthetic:raise ValueError('Baseline manifest must be real-only')
        if role=='S1' and not synthetic:raise ValueError('S1 synthesis branch requires reviewed synthetic training data')


def build_model(tf,architecture,seed,backbone_weights="imagenet"):
    layers=tf.keras.layers
    augmentation=tf.keras.Sequential([layers.RandomFlip('horizontal',seed=seed),layers.RandomRotation(.1,seed=seed+1),layers.RandomZoom(.1,seed=seed+2),layers.RandomTranslation(.1,.1,seed=seed+3)],name='augmentation')
    if architecture=='baseline':
        return tf.keras.Sequential([layers.Input((224,224,3)),augmentation,layers.Rescaling(1/255),
            layers.Conv2D(32,3,activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64,3,activation='relu'),layers.MaxPooling2D(),
            layers.Conv2D(128,3,activation='relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128,activation='relu'),
            layers.Dropout(.5),layers.Dense(3,activation='softmax')],name='baseline_cnn'),None
    if architecture=='convnext_tiny':
        base=tf.keras.applications.ConvNeXtTiny(input_shape=(224,224,3),include_top=False,
            weights=backbone_weights,pooling=None,include_preprocessing=True)
        base.trainable=False
        # Explicit independent head seed keeps A0/S1 head initialization identical.
        head=layers.Dense(3,activation='softmax',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+100),bias_initializer='zeros')
        return tf.keras.Sequential([layers.Input((224,224,3)),base,layers.GlobalAveragePooling2D(),head],name='convnext_gestures'),base
    base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
    base.trainable=False
    return tf.keras.Sequential([layers.Input((224,224,3)),augmentation,layers.Rescaling(1/127.5,offset=-1),base,
        layers.GlobalAveragePooling2D(),layers.Dropout(.3),layers.Dense(3,activation='softmax')],name='mobilenetv2'),base


def train(config_path,root):
    root=Path(root).resolve();config,manifest,run=preflight(config_path,root)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(config['seed'])
    total_started=time.perf_counter()
    model,base=build_model(tf,config['architecture'],config['seed'],config.get('backbone_weights','imagenet'))
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',config);write_json(run/'class_names.json',list(CLASS_NAMES))
    write_json(run/'preprocessing.json',DEFAULT.to_dict())
    model.save(run/'initial.keras')
    def arrays(split):
        rows=[r for r in manifest['rows'] if r['split']==split and r['review_status']=='accepted']
        x=np.concatenate([model_input(open_rgb(root/r['path']),already_cropped=r['already_cropped']) for r in rows])
        y=np.array([CLASS_NAMES.index(r['label']) for r in rows],dtype=np.int32)
        return rows,x,y
    train_rows,x,y=arrays('train');val_rows,vx,vy=arrays('val')
    write_json(run/'sample_ids.json',{'train':[r['sample_id'] for r in train_rows],'val':[r['sample_id'] for r in val_rows]})
    fit_model=model
    feature_seconds=0.0
    if config['architecture']=='convnext_tiny':
        # Frozen deterministic representation: cache only train/val, never test.
        feature_start=time.perf_counter()
        def extract(images):
            chunks=[]
            for start in range(0,len(images),8):
                spatial=base(images[start:start+8],training=False)
                chunks.append(tf.reduce_mean(spatial,axis=(1,2)).numpy())
            return np.concatenate(chunks)
        x=extract(x);vx=extract(vx)
        feature_seconds=time.perf_counter()-feature_start
        fit_model=tf.keras.Sequential([tf.keras.layers.Input((768,)),model.layers[-1]])
        np.savez_compressed(run/'features_train_val.npz',train=x,val=vx)
    class FullCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self,path):super().__init__();self.path=path;self.best=-float('inf')
        def on_epoch_end(self,epoch,logs=None):
            value=(logs or {}).get('val_accuracy',-float('inf'))
            if value>self.best:self.best=value;model.save(self.path)
    ds=tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(y),seed=config['seed'],reshuffle_each_iteration=True).repeat().batch(config['batch_size']).prefetch(1)
    options=tf.data.Options();options.threading.private_threadpool_size=2
    ds=ds.with_options(options)
    validation=tf.data.Dataset.from_tensor_slices((vx,vy)).batch(config['batch_size']).with_options(options)
    phases=[('frozen' if base else 'baseline',config['epochs'],config['learning_rate'])]
    if base and config['fine_tune_epochs']>0:phases.append(('finetune',config['fine_tune_epochs'],config['fine_tune_learning_rate']))
    histories={};started=time.perf_counter();write_json(run/'state.json',{'status':'RUNNING'})
    try:
        for phase,epochs,lr in phases:
            if phase=='finetune':
                base.trainable=True
                for i,layer in enumerate(base.layers):
                    layer.trainable=i>=len(base.layers)-30 and not isinstance(layer,tf.keras.layers.BatchNormalization)
            fit_model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            callbacks=[tf.keras.callbacks.CSVLogger(str(run/f'{phase}_epochs.csv')),
                FullCheckpoint(run/f'{phase}_best.keras'),
                tf.keras.callbacks.TerminateOnNaN()]
            history=fit_model.fit(ds,steps_per_epoch=config['steps_per_epoch'],validation_data=validation,epochs=epochs,callbacks=callbacks)
            histories[phase]=history.history
            if any(not np.isfinite(values).all() for values in history.history.values()):raise RuntimeError('Non-finite history')
            if len(history.history['loss'])!=epochs:raise RuntimeError('Training ended before planned epochs')
            write_json(run/f'{phase}_history.json',history.history);model.save(run/f'{phase}_last.keras')
        # Frozen contract uses highest validation accuracy; earliest phase/epoch breaks ties.
        phase=max(histories,key=lambda p:max(histories[p]['val_accuracy']))
        import shutil
        shutil.copy2(run/f'{phase}_best.keras',run/'best.keras')
        write_json(run/'summary.json',{'selected_phase':phase,'selection':'maximum val_accuracy; earliest phase/epoch on ties',
            'parameters':model.count_params(),'training_seconds':time.perf_counter()-total_started,'feature_extraction_seconds':feature_seconds,'weights_sha256':sha256(run/'best.keras'),
            'tensorflow':tf.__version__,'manifest_sha256':config['manifest_sha256'],'status':'TRAINED_NOT_TEST_EVALUATED'})
        write_json(run/'state.json',{'status':'TRAINED_NOT_TEST_EVALUATED'})
    except Exception as exc:
        write_json(run/'state.json',{'status':'FAILED','error':str(exc)})
        raise
    return run
