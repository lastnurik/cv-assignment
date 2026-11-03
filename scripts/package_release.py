"""Explicit allowlist: source export separate from private course-review assets."""
import json,shutil,subprocess,zipfile
from pathlib import Path
from gesture_cv.dataset import write_json,sha256

def main():
    target=Path('deliverables/repository')
    if target.exists():raise FileExistsError('Release export already exists; inspect before rebuilding')
    files=[Path(x) for x in ['README.md','requirements-runtime.lock.txt','requirements-runtime.in','demo.py','train.py','collect_data.py','deliverables/model_bundle.json','deliverables/RELEASE_GUIDE.md','data/manifests/a1_limited_v2.json','configs/experiment_contract.convnext_v1.json']]
    files+=list(Path('gesture_cv').glob('*.py'))+list(Path('web').glob('*'))+list(Path('tests').glob('*.py'))
    files += [Path('scripts')/x for x in ['validate_data.py','evaluate_predictions.py','plot_release.py']]
    files += list(Path('configs').glob('*.convnext_v1.json'))
    files += [Path('artifacts/evaluation')/x for x in ['comparison.json','selection.json','verification.json','latency.json','confusion_matrices.png','training_curves.png','loss_curves.png']]
    files += [Path('artifacts/method/feasible_alternative.md')]
    for role in ['B0','B1','S1','A0']:
        run=Path('runs')/(role+'_limited_v1');files += list(run.glob('*.json'))+list(run.glob('*epochs.csv'))
    for f in sorted(set(files)):
        dest=target/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    (target/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\n*.keras\n*.h5\ndata/clean/\ndataset/\nartifacts/method/pilot_keras_cache/\n*.zip\n*.mp4\n')
    # Tests referencing intentionally-unfrozen example guards need their fixtures.
    for name in ['run.example.json','experiment_contract.draft.json']:
        shutil.copy2(Path('configs')/name,target/'configs'/name)
    subprocess.run(['git','init',str(target)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(target),'add','.'],check=True)
    subprocess.run(['git','-C',str(target),'-c','user.name=Gesture Lab','-c','user.email=gesture-lab@local.invalid','commit','-m','Assignment 2 verified local release'],check=True,capture_output=True)
    commit=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    with zipfile.ZipFile('deliverables/A2_Source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in target.rglob('*'):
            if f.is_file():z.write(f,str(f.relative_to(target)))
    private=list(Path('dataset').rglob('*.jpg'))+list(Path('data/clean/a1_clean_v1').rglob('*.png'))
    private += [Path('runs')/(r+'_limited_v1')/'best.keras' for r in ['B0','B1','S1','A0']]
    with zipfile.ZipFile('deliverables/A2_Private_Assets.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in private:z.write(f,str(f))
    write_json('artifacts/qa/release_package.json',{'source_commit':commit,'source_files':len(set(files)),'private_files':len(private),'source_zip_sha256':sha256('deliverables/A2_Source.zip'),'private_zip_sha256':sha256('deliverables/A2_Private_Assets.zip'),'public_source_excludes_images_and_weights':True,'published':False})
    print('Packaged local repository',commit)
if __name__=='__main__':main()
