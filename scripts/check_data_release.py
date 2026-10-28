"""Verify every required artifact from the immutable public data release."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
release = json.loads((ROOT / 'ml4hl/assets/data-release.json').read_text())
cache = ROOT / '.cache/release-verification'
os.environ.setdefault('HF_XET_CACHE', str(ROOT / '.cache/xet'))
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
from huggingface_hub import hf_hub_download

verified = []
for name, expected in release['files'].items():
    path = Path(hf_hub_download(release['repo'], 'artifacts/' + name,
                               repo_type='dataset', revision=release['revision'],
                               cache_dir=str(cache), token=False))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, f'Release checksum mismatch: {name}'
    verified.append(dict(path='artifacts/' + name, sha256=actual, match=True))
receipt = dict(release=release, files=verified)
(ROOT / '.cache/data-release-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(f'PASS: {len(verified)} required public artifacts at {release["revision"]}')
