"""Write a content-addressed helper archive and a compact notebook downloader."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile


def helper_archive(root: Path) -> tuple[str, bytes]:
    """Deterministically pack ml4hl/ into ZIP bytes and return (sha256, payload)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root / "ml4hl").rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
                and path.name != ".DS_Store"
            ):
                info = zipfile.ZipInfo(str(path.relative_to(root)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    payload = buf.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    return checksum, payload


def bundled_helper_setup(root: Path) -> str:
    checksum, payload = helper_archive(root)
    destination = root / "assets" / "helpers" / f"ml4hl-{checksum}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return f'''

# Download this notebook's reviewed helper version (Python, JavaScript, and CSS).
import hashlib, io, tempfile, zipfile, os
from pathlib import Path
from urllib.request import urlopen
os.environ["ML4HL_STANDALONE"] = "1"
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = os.environ.get("ML4HL_THREADS", "2")
_helper_sha256 = {checksum!r}
_helper_url = (
    "https://raw.githubusercontent.com/ColinConwell/ML4HL-Workshop/main/"
    f"assets/helpers/ml4hl-{{_helper_sha256}}.zip"
)
_local_helper = Path(f"assets/helpers/ml4hl-{{_helper_sha256}}.zip")
if _local_helper.is_file():
    _helper_zip = _local_helper.read_bytes()
else:
    with urlopen(_helper_url, timeout=60) as _response:
        _helper_zip = _response.read()
if hashlib.sha256(_helper_zip).hexdigest() != _helper_sha256:
    raise RuntimeError("Helper download failed its SHA-256 check; nothing was loaded.")
_helper_dir = tempfile.mkdtemp(prefix="ml4hl-helper-")
with zipfile.ZipFile(io.BytesIO(_helper_zip)) as _archive:
    _archive.extractall(_helper_dir)
sys.path.insert(0, _helper_dir)
for _name in list(sys.modules):
    if _name == "ml4hl" or _name.startswith("ml4hl."):
        del sys.modules[_name]
import sklearn
from packaging.version import Version
if Version(sklearn.__version__) < Version("1.6"):
    raise RuntimeError("This notebook needs scikit-learn >= 1.6. Upgrade it and restart the runtime.")
import ml4hl
from IPython.display import HTML, display
display(HTML(ml4hl.theming.theme_html()))
'''
