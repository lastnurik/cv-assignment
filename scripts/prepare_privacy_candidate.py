"""Uniform privacy-crop feasibility candidate, never auto-approve samples."""
import json
from pathlib import Path
from gesture_cv.dataset import write_json,sha256,contact_sheets
from gesture_cv.preprocessing import open_rgb

def main():
    manifest=json.loads(Path('data/manifests/a1_clean_v1.json').read_text())
    version='a1_privacy_candidate_v1';out=Path('data/clean')/version
    target_manifest=Path('data/manifests')/(version+'.json')
    if out.exists() or target_manifest.exists():raise FileExistsError('Version exists')
    for row in manifest['rows']:
        im=open_rgb(row['source_path']);assert im.size==(640,480)
        target=out/row['label']/(row['sample_id']+'.png');target.parent.mkdir(parents=True,exist_ok=True)
        cropped=im.crop((0,96,480,480))
        with target.open('xb') as f:cropped.save(f,format='PNG')
        assert open_rgb(target).tobytes()==cropped.tobytes()
        row.update(path=str(target),sha256=sha256(target),crop_box=[0,96,480,480],
          transform={'crop_top_fraction':.2,'crop_right_fraction':.25,'width':224,'height':224,'version':'privacy_top20_right25_v1'},
          review_status='pending',review_reason='New crop requires fresh privacy/gesture review',
          participant_id='teammate_p01',participant_consent_status='private_use_confirmed',consent_status='unknown')
    manifest.update(version=version,status='PRIVACY_CROP_REVIEW_PENDING',notes='Candidate only; production preprocessing unchanged; not ready for training')
    write_json(target_manifest,manifest)
    print(contact_sheets('.',manifest))
if __name__=='__main__':main()
