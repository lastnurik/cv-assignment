"""Fresh isolated env/export inference check; private fixture only, server auto-stops."""
import json,shutil,subprocess,tempfile,time,urllib.request
from pathlib import Path
from gesture_cv.dataset import write_json,sha256

def main():
    with tempfile.TemporaryDirectory(prefix='gesture-release-') as td:
        root=Path(td);checkout=root/'project';shutil.copytree('deliverables/repository',checkout)
        weights=Path('runs/S1_limited_v1/best.keras');(checkout/weights).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(weights,checkout/weights)
        py=root/'venv/bin/python'
        with open('artifacts/qa/fresh_checkout.log','w') as log:
            subprocess.run(['uv','venv','--python','3.12',str(root/'venv')],check=True,stdout=log,stderr=log)
            subprocess.run(['uv','pip','install','--python',str(py),'-r',str(checkout/'requirements-runtime.lock.txt')],check=True,stdout=log,stderr=log)
            subprocess.run([str(py),'-m','unittest','discover','-s','tests','-v'],cwd=checkout,check=True,stdout=log,stderr=log)
            server=subprocess.Popen([str(py),'demo.py','--port','8766','--bundle','deliverables/model_bundle.json'],cwd=checkout,stdout=log,stderr=log)
            try:
                for _ in range(60):
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8766/api/status',timeout=2) as r:status=json.load(r)
                        break
                    except Exception:time.sleep(.5)
                else:raise RuntimeError('Fresh server failed startup')
                req=urllib.request.Request('http://127.0.0.1:8766/api/predict',data=Path('dataset/peace/peace_0000.jpg').read_bytes())
                with urllib.request.urlopen(req,timeout=90) as r:prediction=json.load(r)
                assert prediction['prediction']=='peace' and prediction['explanation_valid']
            finally:server.terminate();server.wait(timeout=10)
        write_json('artifacts/qa/fresh_checkout.json',{'passed':True,'new_environment':True,'exported_source':True,'unit_tests_passed':True,'actual_http_prediction':prediction['prediction'],'actual_gradcam_nonzero':prediction['explanation_valid'],'weights_sha256':sha256(weights),'scope':'Fresh pinned CPU runtime + exported source + explicitly provided selected weights and one private fixture; no hidden project imports','server_stopped':True})
if __name__=='__main__':main()
