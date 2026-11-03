"""A2 training: requires a frozen experiment contract and reviewed split manifest.
Original A1 source is preserved in artifacts/a1_original/train.py.
"""
import argparse
from pathlib import Path
from gesture_cv.training import preflight,train

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    try:
        if a.check_only:
            preflight(a.config,a.root);print('Training preflight passed; no training performed.')
        else:print(train(a.config,a.root))
    except (ValueError,FileNotFoundError,ImportError) as exc:
        raise SystemExit(str(exc)) from exc
if __name__=='__main__':main()
