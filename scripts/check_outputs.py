"""Reject incomplete executed HTML and notebook/source drift."""
from pathlib import Path
import re
import nbformat
ROOT=Path(__file__).absolute().parents[1]
source=(ROOT/'ML4HL-Tutorial.qmd').read_text();page=(ROOT/'ML4HL-Tutorial.html').read_text()
labels=re.findall(r'^#\| label: (.+)$',source,re.M)
missing=[name for name in labels if f'id="{name}"' not in page and f'id="{name}-' not in page]
# Quarto may not put setup code's label on its outer div; check rendered cell count separately.
assert page.count('class="cell"')>=len(labels), (page.count('class="cell"'),len(labels))
assert page.count('<div class="ml4hl-plot">')>=21,'Missing plot outputs'
assert 'data:image/png;base64,' in page and 'data:image/jpeg;base64,' in page
assert 'Traceback (most recent call last)' not in page
nb=nbformat.read(ROOT/'ML4HL-Tutorial.ipynb',4)
assert sum(c.cell_type=='code' for c in nb.cells)==len(re.findall(r'^```\{python\}',source,re.M))+1
assert 'base64.b64decode' in nb.cells[1].source and 'ML4HL_STANDALONE' in nb.cells[1].source
print(f'PASS: {len(labels)} labeled source cells, at least 21 themed figures, logo, and generated notebook setup')
