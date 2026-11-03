import argparse,json
from pathlib import Path
from gesture_cv.metrics import compare_aligned
from gesture_cv.dataset import write_json

def main():
    p=argparse.ArgumentParser(description='Recompute comparison from aligned real saved predictions; no fabricated results.')
    p.add_argument('predictions',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();runs={}
    for path in a.predictions:
        if path.parent.name in runs:p.error('Run directories must have distinct names')
        runs[path.parent.name]=json.loads(path.read_text())
    write_json(a.output,compare_aligned(runs))
if __name__=='__main__':main()
