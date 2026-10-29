"""Reject incomplete executed HTML and notebook/source drift."""
from pathlib import Path
import ast
import hashlib
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
assert 'urlopen(_helper_url' in nb.cells[1].source and 'ML4HL_STANDALONE' in nb.cells[1].source
assert 'hashlib.sha256(_helper_zip)' in nb.cells[1].source
assert max(map(len, nb.cells[1].source.splitlines())) < 200, 'Oversized notebook setup line'
setup = ast.parse(nb.cells[1].source)
helper_hash = next(node.value.value for node in setup.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '_helper_sha256' for target in node.targets))
assert re.fullmatch(r'[a-f0-9]{64}', helper_hash)
helper = ROOT / 'assets' / 'helpers' / f'ml4hl-{helper_hash}.zip'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == helper_hash, 'Missing or altered helper archive'
print(f'PASS: {len(labels)} labeled source cells, at least 21 themed figures, logo, and generated notebook setup')

# Publication requirements shared by Quarto and the generated Colab notebook.
blocks = re.findall(r"```\{python\}\n(.*?)\n```", source, re.S)
figure_blocks = [block for block in blocks if "plt.subplots(" in block]
assert all("#| code-fold: true" in block for block in figure_blocks), "Unfolded plotting cell"
assert all(not re.search(r"\.fit(?:_transform)?\(|cross_val_(?:score|predict)\(", block) for block in figure_blocks), "Model fitting remains in plotting code"
figure_cells = [cell for cell in nb.cells if cell.cell_type == "code" and "plt.subplots(" in cell.source]
assert len(figure_cells) == len(figure_blocks)
assert all(cell.metadata.get("cellView") == "form" for cell in figure_cells), "Colab plot folding lost"
takeaways = re.findall(r'title="Takeaway #\d+"}\n(.*?)\n:::', source, re.S)
closing = source.split("## The Ten Takeaways\n")[1].split("## The Whole Toolkit")[0]
assert len(takeaways) == 10
assert all(f"{i}. {text}" in closing for i, text in enumerate(takeaways, 1)), "Closing takeaways drifted"
for content in (source, page, "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")):
 assert not re.search(r"\bPart (?:2|II)\b|optional local widget gallery", content, re.I), "Retired public reference remains"
print(f"PASS: {len(figure_blocks)} folded plot cells in HTML source and Colab; separate fitting; ten synchronized takeaways; retired references removed")

stack = []
for line in source.splitlines():
 if re.match(r"^:::+\s+\{", line): stack.append(line)
 elif re.match(r"^:::+\s*$", line):
  assert stack, "Closing a Markdown container that is not open"
  stack.pop()
assert not stack, f"Unclosed Markdown containers: {stack}"

widget_blocks = [block for block in blocks if re.search(r"^show\(", block, re.M)]
assert all("#| code-fold: true" in block for block in widget_blocks), "Interactive figure code is not expandable and folded"
print(f"PASS: {len(widget_blocks)} interactive figure cells are folded and expandable")
