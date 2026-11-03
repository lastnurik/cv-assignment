"""Apply recorded agent visual review of all six a1_clean_v1 contact sheets.
No consent or independent-session inference is made by this review.
"""
import json
from pathlib import Path
from collections import Counter
from gesture_cv.dataset import write_json

def main():
    path=Path('data/manifests/a1_clean_v1.json');manifest=json.loads(path.read_text())
    records=[]
    for row in manifest['rows']:
        no_gesture=row['label']=='stop' and int(row['sample_id'].rsplit('_',1)[1])>=152
        row['review_status']='rejected' if no_gesture else 'accepted'
        row['review_reason']='No visible stop gesture; do not relabel as an existing class.' if no_gesture else 'Gesture consistent with label in six-sheet visual review; overlay removed. Some natural blur/occlusion retained.'
        row['reviewer']='primary_agent_visual_review_2026-11-04'
        # Environment is a visual descriptor, not independent-session evidence.
        row['environment']='original_indoor_room'
        records.append({k:row[k] for k in ['sample_id','review_status','review_reason','reviewer','sha256']})
    manifest['status']='VISUALLY_REVIEWED_UNSPLIT_CONSENT_PENDING'
    write_json(path,manifest)
    write_json('artifacts/audit/visual_review.json',{'coverage':'All six contact sheets, all 464 images; stop_0152, thumbs_up_0067, peace_0037 additionally inspected full resolution',
        'crop':'top 20%, y=96 for all 640x480 originals; no readable collection overlay remaining',
        'limitations':'One apparent setting and participant; background bystanders; consent unconfirmed; clipped fingertips and blur retained where gesture remains recognizable.',
        'records':records})
    print(Counter((r['label'],r['review_status']) for r in manifest['rows']))
if __name__=='__main__':main()
