"""Bundle reviewed helper sources into the generated notebook's setup cell."""
import base64
import hashlib
import io
import zipfile


def bundled_helper_setup(root):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root / 'ml4hl').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                info = zipfile.ZipInfo(str(path.relative_to(root)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    payload = buf.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    return f'''

# Reviewed ml4hl source; bundled by scripts/to_colab.py. SHA-256: {checksum}
import base64, hashlib, io, tempfile, zipfile, os
os.environ["ML4HL_STANDALONE"] = "1"
_helper_zip = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(_helper_zip).hexdigest() == {checksum!r}
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
