"""Bundle the selected theme and VISNA logo for self-contained Quarto output."""
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from ml4hl.theming import theme_html

logo = 'data:image/jpeg;base64,' + base64.b64encode((ROOT/'assets/visna-logo.jpg').read_bytes()).decode()
(ROOT/'assets/theme-head.html').write_text('<script>window.ML4HL_LOGO=' + repr(logo) + ';</script>' + theme_html())
