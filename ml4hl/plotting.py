"""Matplotlib / seaborn styling that matches the interactive widgets."""

from __future__ import annotations

PALETTE = ["#2563eb", "#d97706", "#16a34a", "#b83a2a", "#7c3aed", "#0f766e",
           "#db2777", "#4d7c0f", "#0369a1", "#a16207", "#6b7280", "#9333ea"]
INK, MUTED, GRID, BG = "#2a2a2a", "#8a877e", "#e7e3da", "#faf8f4"
COLORS = dict(blue="#2563eb", orange="#d97706", green="#16a34a", red="#b83a2a", purple="#7c3aed",
              teal="#0f766e", gold="#ca8a04", ink=INK, muted=MUTED, grid=GRID, bg=BG)


def set_style(context: str = "notebook") -> None:
    """Apply the ML4HL look to matplotlib and seaborn figures."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(context=context, style="whitegrid", palette=PALETTE)
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": "#cfc9bc", "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlesize": "medium",
        "grid.color": GRID, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 110,
    })
