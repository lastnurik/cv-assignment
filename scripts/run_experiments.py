"""Sequential controlled runs; stops on first failure, never overwrites a run."""
import os,subprocess,sys,time
from pathlib import Path
from gesture_cv.dataset import write_json,sha256

def main():
    env=dict(os.environ,KERAS_HOME=str(Path('artifacts/method/pilot_keras_cache').resolve()),MPLCONFIGDIR='/tmp/cv-matplotlib')
    roles=['B0','B1','S1','A0'];records=[]
    for role in roles:
        command=[sys.executable,'train.py','--config',f'configs/{role}.convnext_v1.json']
        state={'status':'RUNNING','runner_pid':os.getpid(),'active_role':role,'command':command,'completed':records,'start_time':time.time()}
        write_json('artifacts/qa/experiment_jobs.json',state)
        with open(f'artifacts/qa/{role}_limited_v1.log','x') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append({'role':role,'exit_code':proc.returncode})
        if proc.returncode:
            write_json('artifacts/qa/experiment_jobs.json',dict(state,status='FAILED',completed=records));raise SystemExit(proc.returncode)
    write_json('artifacts/qa/experiment_jobs.json',{'status':'TRAINING_FINISHED_PENDING_REVIEW','completed':records,'runner_pid':os.getpid(),'source_sha256':sha256('gesture_cv/training.py')})
if __name__=='__main__':main()
