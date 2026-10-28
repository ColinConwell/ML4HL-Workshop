"""Assemble self-contained HTML for the ML-4-HL JavaScript widgets.

Every call to :func:`show` inlines the widget stylesheet, the shared ``core.js``
runtime, the widget's own script, and its JSON config. Colab renders each cell
output in its own sandboxed iframe, so nothing can be shared between outputs.
Quarto, JupyterLab, and VS Code render outputs into one document, so there the
shared assets are emitted once and later widgets reuse them.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from functools import lru_cache
from pathlib import Path
from .theming import theme_html
from typing import Any

import numpy as np

ASSETS = Path(__file__).absolute().parent / "assets"  # not resolve(): Hub snapshots are symlinks into blobs/

FONT_LINK = ""  # System fonts keep standalone outputs offline.


_EMITTED: set[str] = set()


@lru_cache(maxsize=None)
def _read(rel: str) -> str:
    return (ASSETS / rel).read_text(encoding="utf-8")


def manifest() -> dict:
    """The widget manifest (figure numbers, titles, parameters, data modes)."""
    return json.loads(_read("widgets.json"))


def list_widgets() -> list[str]:
    return [w["name"] for w in manifest()["widgets"]]


def _spec(name: str) -> dict:
    for w in manifest()["widgets"]:
        if w["name"] == name:
            return w
    raise KeyError(f"Unknown widget {name!r}. Available: {', '.join(list_widgets())}")


def _isolated_outputs() -> bool:
    """True where every cell output is its own iframe (Colab), so assets must repeat."""
    flag = os.environ.get("ML4HL_STANDALONE")
    if flag is not None:
        return flag.strip().lower() not in ("0", "false", "no", "")
    import sys

    return "google.colab" in sys.modules


def to_jsonable(obj: Any, digits: int = 4) -> Any:
    """Convert numpy / pandas containers into compact JSON-ready Python objects."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None

    if pd is not None and isinstance(obj, pd.DataFrame):
        return {str(c): to_jsonable(obj[c].to_numpy(), digits) for c in obj.columns}
    if pd is not None and isinstance(obj, (pd.Series, pd.Index)):
        return to_jsonable(obj.to_numpy(), digits)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "f":
            arr = np.round(obj.astype(np.float64), digits)
            return [to_jsonable(v, digits) for v in arr] if arr.ndim > 1 else [
                None if not math.isfinite(v) else (int(v) if v.is_integer() else v)
                for v in arr.tolist()
            ]
        if obj.dtype.kind in "iub":
            return obj.tolist()
        return [to_jsonable(v, digits) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item(), digits)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        r = round(obj, digits)
        return int(r) if r.is_integer() else r
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, digits) for v in obj]
    return obj


def _script_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def widget_html(
    name: str,
    data: Any = None,
    *,
    digits: int = 4,
    standalone: bool | None = None,
    **params: Any,
) -> str:
    """Return the HTML for one widget.

    Parameters
    ----------
    name : widget name (see :func:`list_widgets`).
    data : optional real data for the widget's data mode (arrays, DataFrames, dicts).
    standalone : inline every asset in this output (default: only in Colab, or
        when ``ML4HL_STANDALONE=1``). Elsewhere (Quarto, Jupyter, VS Code) the
        page is shared, so the CSS and runtime are emitted once.
    **params : widget configuration (``seed``, ``n``, ``noise``, ``title`` ...).
    """
    spec = _spec(name)
    if standalone is None:
        standalone = _isolated_outputs()

    cfg = {"fig": spec.get("fig"), "title": spec.get("title"), "hint": spec.get("hint")}
    cfg.update(params)
    if data is not None:
        cfg["data"] = data
    cfg = to_jsonable(cfg, digits)

    uid = f"ml4hl-{name}-{uuid.uuid4().hex[:8]}"
    parts: list[str] = []
    if standalone or "theme" not in _EMITTED:
        parts.append(theme_html())
        _EMITTED.add("theme")
    if standalone:
        parts += [FONT_LINK, f"<style>{_read('ml4hl-widgets.css')}</style>",
                  f"<script>{_read('core.js')}</script>",
                  f"<script>{_read(f'widgets/{name}.js')}</script>"]
    else:
        for key, chunk in (("css", f"<style>{_read('ml4hl-widgets.css')}</style>"),
                           ("core", f"<script>{_read('core.js')}</script>"),
                           (name, f"<script>{_read(f'widgets/{name}.js')}</script>")):
            if key not in _EMITTED:
                parts.append(chunk)
                _EMITTED.add(key)
    parts.append(f'<div id="{uid}" class="ml4hl-root" data-widget="{name}"></div>')
    parts.append(
        f'<script>if (window.ML4HL) ML4HL.mount("{name}", "{uid}", {_script_json(cfg)}); '
        f'else document.getElementById("{uid}").textContent = '
        '"ml4hl: widget runtime missing. Re-run the setup cell (it calls ml4hl.reset_assets()).";</script>'
    )
    return "\n".join(parts)


def reset_assets() -> None:
    """Forget which shared assets were emitted (call once per fresh page / render)."""
    _EMITTED.clear()


def show(name: str, data: Any = None, **params: Any):
    """Display an interactive widget in Jupyter, Colab, VS Code, or a Quarto render."""
    from IPython.display import HTML

    return HTML(widget_html(name, data, **params))
