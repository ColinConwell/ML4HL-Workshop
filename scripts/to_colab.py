"""Convert ML4HL-Tutorial.qmd into a Colab-ready notebook (ML4HL-Tutorial.ipynb).

Quarto-only markdown is rewritten into plain Markdown that Colab renders:
callouts become blockquotes, tabsets become ordinary subsections, the HTML
roadmap / field cards become bullet lists, and definition lists become bullets.
Widget-only cells (``#| echo: false``) become collapsed Colab form cells, and a
setup cell installs the few packages the Colab runtime does not ship.

Usage:
    python scripts/to_colab.py [--src ML4HL-Tutorial.qmd] [--out ML4HL-Tutorial.ipynb]
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import re
from pathlib import Path

import nbformat
from notebook_bundle import bundled_helper_setup
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
GH_TREE = "https://colinconwell.com/Tutorials/tutorials"
COLAB_URL = "https://colab.research.google.com/github/ColinConwell/ML4HL-Workshop/blob/main/ML4HL-Tutorial.ipynb"
COLAB_PACKAGES = ["rsatoolbox", "medmnist", "huggingface_hub", "pyarrow", "seaborn"]

SETUP = f'''# @title In Colab? Click Play
import importlib.util, subprocess, sys

missing = [p for p in {COLAB_PACKAGES!r} if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)'''


def front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        if v.strip() and not line.startswith(" "):
            meta[k.strip()] = v.strip().strip('"')
    return meta, text[m.end():]


def split_cells(body: str) -> list[tuple[str, str]]:
    cells, buf, code, kind = [], [], False, "markdown"
    for line in body.splitlines():
        if not code and re.match(r"^```\{python\}", line):
            cells.append((kind, "\n".join(buf))); buf, code, kind = [], True, "code"
        elif code and line.strip() == "```":
            cells.append((kind, "\n".join(buf))); buf, code, kind = [], False, "markdown"
        else:
            buf.append(line)
    cells.append((kind, "\n".join(buf)))
    return [(k, s.strip("\n")) for k, s in cells if s.strip()]


def html_items(block: str) -> list[str]:
    items = []
    for div in re.findall(r"<div[^>]*>(.*?)</div>", block, re.S):
        title = re.search(r"<(?:strong|h4)>(.*?)</(?:strong|h4)>", div)
        desc = re.search(r"<(?:small|p)>(.*?)</(?:small|p)>", div)
        if title:
            items.append(f"- **{title.group(1)}**" + (f": {desc.group(1)}" if desc else ""))
    return items


def convert_markdown(md: str) -> str:
    lines, out, stack, i = md.splitlines(), [], [], 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^:::+\s*\{(.*)\}\s*$", line)
        if m:
            attrs = m.group(1)
            if ".roadmap" in attrs or ".field-grid" in attrs:
                j = i + 1
                while not re.match(r"^:::+\s*$", lines[j]):
                    j += 1
                out += html_items("\n".join(lines[i + 1:j]))
                i = j + 1
                continue
            title = re.search(r'title="([^"]*)"', attrs)
            if ".callout" in attrs:
                stack.append("callout")
                out.append(f"> **{title.group(1)}**" if title else ">")
                out.append(">")
            else:
                stack.append("plain")
            i += 1
            continue
        if re.match(r"^:::+\s*$", line):
            stack.pop() if stack else None
            i += 1
            continue
        if stack and stack[-1] == "callout":
            out.append(f"> {line}".rstrip())
            i += 1
            continue
        if i + 1 < len(lines) and lines[i + 1].startswith(": ") and line.strip() and not line.startswith(("#", "-", "|")):
            out.append(f"- **{line.strip()}**: {lines[i + 1][2:].strip()}")
            i += 2
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    text = re.sub(r"\]\(\.\./([\w-]+)/?\)", rf"]({GH_TREE}/\1)", text)
    text = re.sub(r"\{\.[\w-]+\}", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def convert_code(src: str) -> tuple[str, dict]:
    opts, body = {}, []
    for line in src.splitlines():
        m = re.match(r"^#\|\s*([\w-]+):\s*(.*)$", line)
        if m:
            opts[m.group(1)] = m.group(2).strip().strip('"')
        else:
            body.append(line)
    code, meta = "\n".join(body).strip("\n"), {}
    if opts.get("echo") == "false" or opts.get("code-fold") == "true":
        title = opts.get("code-summary") or opts.get("label", "").replace("w-", "widget: ").replace("-", " ")
        code = f"# @title {title}\n{code}"
        meta["cellView"] = "form"
    return code, meta


def build(src: Path, asset_ref: str = "main") -> nbformat.NotebookNode:
    meta, body = front_matter(src.read_text())
    header = [
        '<div style="display:flow-root;">',
        f'<img src="https://raw.githubusercontent.com/ColinConwell/ML4HL-Workshop/{escape(asset_ref, quote=True)}/assets/visna-logo.jpg" '
        'alt="VISNA Lab" width="125" height="125" align="left" '
        'style="float:left;border-radius:26px;margin:0 24px 16px 0;object-fit:contain;">',
        f'<h1 style="margin:0 0 12px;padding:0;">{escape(meta.get("title", src.stem))}</h1>',
    ]
    if meta.get("subtitle"):
        header.append(f'<p style="margin:0;">{escape(meta["subtitle"])}</p>')
    header.extend(['<br clear="all">', '</div>'])
    header = ['\n'.join(header), f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})']
    cells = [new_markdown_cell("\n\n".join(header)), new_code_cell(SETUP + bundled_helper_setup(ROOT), metadata={"cellView": "form"})]
    for kind, text in split_cells(body):
        if kind == "code":
            code, md = convert_code(text)
            cells.append(new_code_cell(code, metadata=md))
        else:
            md = convert_markdown(text)
            if md:
                cells.append(new_markdown_cell(md))
    for i, cell in enumerate(cells):
        cell.id = hashlib.sha256((str(i) + cell.cell_type + cell.source).encode()).hexdigest()[:12]
    nb = new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "ML4HL-Tutorial.qmd")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--asset-ref", help="Public workshop commit for the logo URL; publisher pins this.")
    args = ap.parse_args()
    out = args.out or args.src.with_suffix(".ipynb")
    ref = args.asset_ref
    if ref is None and out.exists():
        match = re.search(r'ML4HL-Workshop/([a-f0-9]{40})/assets/visna-logo.jpg', out.read_text())
        if match: ref = match.group(1)
    nb = build(args.src, ref or "main")
    nbformat.validate(nb)
    nbformat.write(nb, out)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out.name}: {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
