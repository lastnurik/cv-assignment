import argparse,json
from pathlib import Path
from gesture_cv.dataset import validate_manifest,write_json

def main():
    p=argparse.ArgumentParser(description='Reject unsafe/unreviewed/unassigned datasets before final training.')
    p.add_argument('manifest',type=Path);p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--integrity-only',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();r=validate_manifest(json.loads(a.manifest.read_text()),a.root,not a.integrity_only)
    if a.output:write_json(a.output,r)
    print(json.dumps({'passed':r['passed'],'samples':r['samples'],'error_count':len(r['errors']),'first_errors':r['errors'][:8]},indent=2))
    raise SystemExit(0 if r['passed'] else 1)
if __name__=='__main__':main()
