"""Shared Graphite & Spectrum assets and state-preserving Matplotlib variants."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

ASSETS = Path(__file__).absolute().parent / "assets"
THEME = json.loads((ASSETS / "theme.json").read_text())


def theme_html() -> str:
    return ('<style>' + (ASSETS / 'theme.css').read_text() + '</style>'
            '<script>window.ML4HL_THEME_CONFIG=' + json.dumps(THEME) + ';</script>'
            '<script>' + (ASSETS / 'theme.js').read_text() + '</script>')


def figure_html(fig) -> str:
    """Render existing artists three ways, restoring every modified property."""
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    from .plotting import COLORS, PALETTE

    # Only exact house-palette colors are remapped. Images and scientific colormaps
    # are deliberately untouched, including photographs and heatmap cell values.
    source = dict(COLORS, surface='#ffffff', border='#cfc9bc')
    changes = []
    images = []
    alt = '; '.join(ax.get_title() for ax in fig.axes if ax.get_title()) or 'Scientific plot'
    for mode in ('original', 'light', 'dark'):
        target = THEME.get(mode)
        if target:
            mapping = {tuple(to_rgba(v)[:3]): to_rgba(target[k])[:3] for k,v in source.items() if k in target}
            for i, v in enumerate(PALETTE):
                mapping.setdefault(tuple(to_rgba(v)[:3]), to_rgba(target[THEME['categorical'][i % len(THEME['categorical'])]])[:3])
            def recolor(value):
                arr = np.asarray(value) if not isinstance(value, str) else None
                if arr is not None and arr.ndim == 2:
                    return np.array([recolor(v) for v in arr])
                try:
                    rgba = to_rgba(value)
                except (ValueError, TypeError):
                    return value
                rgb = mapping.get(tuple(rgba[:3]))
                return (*rgb, rgba[3]) if rgb else value
            for artist in fig.findobj():
                props = ['color'] if isinstance(artist, (Text, Line2D)) else []
                if isinstance(artist, Line2D): props += ['markerfacecolor', 'markeredgecolor']
                if isinstance(artist, (Patch, Collection)): props += ['facecolor', 'edgecolor']
                for prop in props:
                    getter, setter = getattr(artist, 'get_'+prop), getattr(artist, 'set_'+prop)
                    value = getter()
                    if hasattr(value, 'copy'): value = value.copy()
                    changes.append((setter, value))
                    setter(recolor(value))
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=145, bbox_inches='tight', facecolor=fig.get_facecolor())
            uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
            images.append(f'<img data-mode="{mode}" src="{uri}" alt="{html.escape(alt, quote=True)}">')
        finally:
            for setter, value in reversed(changes): setter(value)
            changes.clear()
    return '<div class="ml4hl-plot">' + ''.join(images) + '</div>'


def enable_theme_plots() -> None:
    """Enable HTML plot variants in an IPython session without recomputing data."""
    from IPython import get_ipython
    from IPython.display import HTML
    from matplotlib_inline import backend_inline
    from matplotlib.figure import Figure
    shell = get_ipython()
    if shell is not None:
        shell.display_formatter.formatters['text/html'].for_type(Figure, figure_html)
        original = getattr(backend_inline, '_ml4hl_original_display', backend_inline.display)
        backend_inline._ml4hl_original_display = original
        def themed_display(obj, *args, **kwargs):
            if isinstance(obj, Figure):
                # Emit HTML explicitly: Quarto otherwise prefers the PNG MIME entry.
                return original(HTML(theme_html() + figure_html(obj)))
            return original(obj, *args, **kwargs)
        backend_inline.display = themed_display
