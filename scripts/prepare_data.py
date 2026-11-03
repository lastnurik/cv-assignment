"""Run as python -m scripts.prepare_data; keeps originals intact."""
import argparse
from pathlib import Path
from gesture_cv.dataset import clean_originals, contact_sheets, validate_manifest, perceptual_candidates, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--version',default='a1_clean_v1')
    args=p.parse_args()
    if not args.version.replace('_','').replace('-','').isalnum():p.error('Use a safe version identifier')
    path,manifest=clean_originals(args.root,args.version)
    check=validate_manifest(manifest,args.root,require_ready=False)
    write_json(args.root/'artifacts/audit'/f'{args.version}_integrity.json',check)
    write_json(args.root/'artifacts/audit'/f'{args.version}_near_duplicates.json',perceptual_candidates(args.root,manifest))
    sheets=contact_sheets(args.root,manifest)
    print(path, '\n',check,'\nSheets:',len(sheets))
    if not check['passed']:raise SystemExit(1)
if __name__=='__main__':main()
