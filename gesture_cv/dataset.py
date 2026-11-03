"""Reproducible cleanup and fail-closed split validation; never invent sessions."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from . import CLASS_NAMES
from .preprocessing import DEFAULT, open_rgb, crop_rgb


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def clean_originals(root, version='a1_clean_v1'):
    root = Path(root).resolve()
    output = root / 'data/clean' / version
    manifest_path = root / 'data/manifests' / f'{version}.json'
    if manifest_path.exists() or output.exists():
        raise FileExistsError('Version already exists; use a new version, never overwrite a reviewed manifest.')
    rows = []
    for label in CLASS_NAMES:
        for source in sorted((root / 'dataset' / label).glob('*.jpg')):
            image = open_rgb(source)
            clean, box = crop_rgb(image)
            target = output / label / (source.stem + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                clean.save(stream, format='PNG')
            rows.append({'sample_id': source.stem, 'source_path': str(source.relative_to(root)),
                         'source_sha256': sha256(source), 'path': str(target.relative_to(root)),
                         'sha256': sha256(target), 'label': label, 'origin': 'real',
                         'session_id': 'unknown', 'group_id': 'a1_original_unknown_session',
                         'participant_id': 'unknown', 'environment': 'unknown',
                         'consent_status': 'unknown', 'split': 'unassigned',
                         'review_status': 'pending', 'review_reason': '',
                         'transform': DEFAULT.to_dict(), 'crop_box': list(box), 'already_cropped': True})
    if not rows:
        raise ValueError('No original dataset images found.')
    manifest = {'version': version, 'status': 'DRAFT_UNSPLIT',
                'class_names': list(CLASS_NAMES), 'rows': rows,
                'notes': 'One conservative original group; session independence/consent unknown. Not ready for training.'}
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def contact_sheets(root, manifest, per_page=80):
    root = Path(root)
    folder = root / 'artifacts/audit/contact_sheets' / manifest['version']
    folder.mkdir(parents=True, exist_ok=True)
    rows = manifest['rows']
    paths = []
    for page, start in enumerate(range(0, len(rows), per_page)):
        items = rows[start:start+per_page]
        sheet = Image.new('RGB', (1600, ((len(items)+7)//8)*145), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(items):
            x, y = (index % 8)*200, (index//8)*145
            with Image.open(root/row['path']) as image:
                thumb = image.copy(); thumb.thumbnail((196,118))
                sheet.paste(thumb, (x,y+20))
            draw.text((x+3,y+3), row['sample_id'], fill='black')
        p = folder / f'{page+1:02d}.jpg';sheet.save(p,quality=90);paths.append(str(p))
    return paths


def validate_manifest(manifest, root, require_ready=True):
    root = Path(root).resolve()
    rows = manifest.get('rows', [])
    errors=[];ids=set();groups=defaultdict(set);hashes=defaultdict(set)
    counts=Counter(); labels=set(CLASS_NAMES)
    if manifest.get('class_names') != list(CLASS_NAMES):errors.append('Class order mismatch')
    if not rows:errors.append('Empty manifest')
    for row in rows:
        sid=row.get('sample_id')
        if not sid or sid in ids:errors.append(f'Duplicate/missing ID: {sid}')
        ids.add(sid)
        if row.get('label') not in labels:errors.append(f'{sid}: invalid label')
        path=(root/row.get('path','')).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f'{sid}: missing/unsafe path');continue
        if sha256(path) != row.get('sha256'):errors.append(f'{sid}: hash mismatch')
        try:
            with Image.open(path) as im: im.load()
        except Exception:errors.append(f'{sid}: invalid image')
        if not require_ready:continue
        if row.get('review_status')=='rejected':continue
        if row.get('review_status')!='accepted':errors.append(f'{sid}: review pending')
        if row.get('consent_status') not in ('private_use_confirmed','public_release_confirmed','not_applicable'):
            errors.append(f'{sid}: consent unresolved')
        split=row.get('split')
        if split not in ('train','val','test'):errors.append(f'{sid}: split unassigned');continue
        group=row.get('group_id')
        if not group:errors.append(f'{sid}: no group ID')
        groups[group].add(split);hashes[row.get('sha256')].add(split)
        # Original and all derivatives must remain together.
        hashes['source:'+str(row.get('source_sha256'))].add(split)
        if row.get('origin') not in ('real','synthetic'):errors.append(f'{sid}: invalid origin')
        if row.get('origin')=='synthetic' and split!='train':errors.append(f'{sid}: synthetic evaluation leakage')
        if split in ('val','test') and row.get('environment') in (None,'unknown',''):
            errors.append(f'{sid}: environment unknown')
        counts[(split,row.get('label'))]+=1
    if require_ready:
        for split in ('train','val','test'):
            for label in CLASS_NAMES:
                if not counts[(split,label)]:errors.append(f'Missing {label} in {split}')
        for group,splits in groups.items():
            if len(splits)>1:errors.append(f'Group leakage: {group}')
        for digest,splits in hashes.items():
            if len(splits)>1:errors.append(f'Duplicate/derivative leakage: {digest}')
    return {'passed':not errors,'errors':errors,'samples':len(rows),
            'counts':{f'{s}/{c}':n for (s,c),n in sorted(counts.items())}}


def perceptual_candidates(root, manifest, threshold=4):
    """64-bit dHash candidates only; similarity is not proof of independence."""
    hashes=[]
    for row in manifest['rows']:
        with Image.open(Path(root)/row['path']) as im:
            pixels=list(im.convert('L').resize((9,8)).get_flattened_data())
        bits=0
        for y in range(8):
            for x in range(8):bits=(bits<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
        hashes.append((row['sample_id'],bits))
    candidates=[]
    for i,(sid,bits) in enumerate(hashes):
        for other,obits in hashes[i+1:]:
            distance=(bits^obits).bit_count()
            if distance<=threshold:candidates.append({'a':sid,'b':other,'distance':distance})
    return {'method':'dHash64','threshold':threshold,'purpose':'review candidates; not automatic relabeling or split assignment','pairs':candidates}
