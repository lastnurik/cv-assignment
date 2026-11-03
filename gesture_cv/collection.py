"""Append-only collection storage, independent of camera/UI dependencies."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import re
from PIL import Image
from . import CLASS_NAMES


def safe_token(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('Use a short alphanumeric identifier with _ or - only.')
    return value


def start_session(root, *, environment, participant='unknown', consent='unknown'):
    safe_token(environment); safe_token(participant)
    if consent not in ('unknown', 'private_use_confirmed', 'public_release_confirmed'):
        raise ValueError('Unknown consent status.')
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:12]
    folder = Path(root) / session
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {'session_id': session, 'environment': environment, 'participant_id': participant,
                'consent_status': consent, 'origin': 'real', 'mirror': True,
                'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (folder / 'session.json').write_text(json.dumps(metadata, indent=2))
    return folder


def save_clean_image(session_folder, label, rgb_image):
    if label not in CLASS_NAMES:
        raise ValueError('Unsupported gesture label.')
    folder = Path(session_folder)
    if not (folder / 'session.json').is_file():
        raise ValueError('Create a collection session first.')
    destination = folder / label
    destination.mkdir(exist_ok=True)
    path = destination / (uuid4().hex + '.png')
    # Exclusive creation prevents overwrites even on an identifier collision.
    try:
        with path.open('xb') as stream:
            rgb_image.convert('RGB').save(stream, format='PNG')
    except Exception:
        # Do not delete an existing file on FileExistsError.
        raise
    return path
