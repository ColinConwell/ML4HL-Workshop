"""Reject incomplete executed HTML and any source-to-artifact drift."""
from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import re
import sys

import nbformat

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).absolute().parent))

from ml4hl.theming import theme_html  # noqa: E402
from notebook_bundle import helper_archive  # noqa: E402


def _norm_css(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


source = (ROOT / "ML4HL-Tutorial.qmd").read_text()
page = (ROOT / "ML4HL-Tutorial.html").read_text()

labels = re.findall(r"^#\| label: (.+)$", source, re.M)
assert page.count('class="cell"') >= len(labels), (page.count('class="cell"'), len(labels))
assert page.count('<div class="ml4hl-plot">') >= 21, "Missing plot outputs"
assert "data:image/png;base64," in page and "data:image/jpeg;base64," in page
assert "Traceback (most recent call last)" not in page
assert "application/vnd.jupyter.widget-view+json" not in page, (
    "Rendered HTML contains transient ipywidgets/tqdm download bars; render with HF_HUB_DISABLE_PROGRESS_BARS=1"
)

# Verify theme-head.html and rendered HTML embed the exact current CSS and JS sources.
logo_uri = "data:image/jpeg;base64," + base64.b64encode((ROOT / "assets/visna-logo.jpg").read_bytes()).decode()
expected_head = "<script>window.ML4HL_LOGO=" + repr(logo_uri) + ";</script>" + theme_html()
assert (ROOT / "assets/theme-head.html").read_text() == expected_head, (
    "assets/theme-head.html drifted from ml4hl/assets/theme.*; run: python scripts/build_theme.py"
)

norm_page = _norm_css(page)
for css_rel in ("styles.css", "ml4hl/assets/theme.css", "ml4hl/assets/ml4hl-widgets.css"):
    assert _norm_css((ROOT / css_rel).read_text()) in norm_page, (
        f"ML4HL-Tutorial.html is stale relative to {css_rel}; re-render ML4HL-Tutorial.qmd"
    )
for js_rel in ("ml4hl/assets/theme.js", "ml4hl/assets/core.js"):
    assert (ROOT / js_rel).read_text() in page, (
        f"ML4HL-Tutorial.html is stale relative to {js_rel}; re-render ML4HL-Tutorial.qmd"
    )
mounted_widgets = sorted(set(re.findall(r'ML4HL\.mount\("([^"]+)"', page)))
assert len(mounted_widgets) >= 19, f"Expected >= 19 mounted widget types, got {len(mounted_widgets)}"
for wname in mounted_widgets:
    wjs = ROOT / "ml4hl/assets/widgets" / f"{wname}.js"
    assert wjs.read_text() in page, (
        f"ML4HL-Tutorial.html is stale relative to {wjs.relative_to(ROOT)}; re-render ML4HL-Tutorial.qmd"
    )

nb = nbformat.read(ROOT / "ML4HL-Tutorial.ipynb", 4)
assert sum(c.cell_type == "code" for c in nb.cells) == len(re.findall(r"^```\{python\}", source, re.M)) + 1
assert "urlopen(_helper_url" in nb.cells[1].source and "ML4HL_STANDALONE" in nb.cells[1].source
assert "hashlib.sha256(_helper_zip)" in nb.cells[1].source
assert max(map(len, nb.cells[1].source.splitlines())) < 200, "Oversized notebook setup line"

setup = ast.parse(nb.cells[1].source)
helper_hash = next(
    node.value.value
    for node in setup.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_helper_sha256" for target in node.targets)
)
assert re.fullmatch(r"[a-f0-9]{64}", helper_hash)
live_helper_hash, _ = helper_archive(ROOT)
assert helper_hash == live_helper_hash, (
    f"ML4HL-Tutorial.ipynb helper hash ({helper_hash[:12]}) drifted from ml4hl/ ({live_helper_hash[:12]}); "
    "run: python scripts/to_colab.py"
)
helper = ROOT / "assets" / "helpers" / f"ml4hl-{helper_hash}.zip"
assert helper.is_file() and hashlib.sha256(helper.read_bytes()).hexdigest() == helper_hash, (
    f"Missing or altered helper archive: {helper.relative_to(ROOT)}; run: python scripts/to_colab.py"
)
print(
    f"PASS: {len(labels)} labeled source cells, at least 21 themed figures, fresh CSS/JS assets, "
    f"and synchronized notebook helper ({helper_hash[:12]})"
)

# Publication requirements shared by Quarto and the generated Colab notebook.
blocks = re.findall(r"```\{python\}\n(.*?)\n```", source, re.S)
figure_blocks = [block for block in blocks if "plt.subplots(" in block]
assert all("#| code-fold: true" in block for block in figure_blocks), "Unfolded plotting cell"
assert all(
    not re.search(r"\.fit(?:_transform)?\(|cross_val_(?:score|predict)\(", block) for block in figure_blocks
), "Model fitting remains in plotting code"
figure_cells = [cell for cell in nb.cells if cell.cell_type == "code" and "plt.subplots(" in cell.source]
assert len(figure_cells) == len(figure_blocks)
assert all(cell.metadata.get("cellView") == "form" for cell in figure_cells), "Colab plot folding lost"
takeaways = re.findall(r'title="Takeaway #\d+"}\n(.*?)\n:::', source, re.S)
closing = source.split("## The Ten Takeaways\n")[1].split("## The Whole Toolkit")[0]
assert len(takeaways) == 10
assert all(f"{i}. {text}" in closing for i, text in enumerate(takeaways, 1)), "Closing takeaways drifted"
for content in (source, page, "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")):
    assert not re.search(r"\bPart (?:2|II)\b|optional local widget gallery", content, re.I), (
        "Retired public reference remains"
    )
print(
    f"PASS: {len(figure_blocks)} folded plot cells in HTML source and Colab; separate fitting; "
    "ten synchronized takeaways; retired references removed"
)

stack = []
for line in source.splitlines():
    if re.match(r"^:::+\s+\{", line):
        stack.append(line)
    elif re.match(r"^:::+\s*$", line):
        assert stack, "Closing a Markdown container that is not open"
        stack.pop()
assert not stack, f"Unclosed Markdown containers: {stack}"

widget_blocks = [block for block in blocks if re.search(r"^show\(", block, re.M)]
assert all("#| code-fold: true" in block for block in widget_blocks), (
    "Interactive figure code is not expandable and folded"
)
print(f"PASS: {len(widget_blocks)} interactive figure cells are folded and expandable")
