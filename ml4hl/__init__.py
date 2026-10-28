"""ML-4-HL: interactive widgets, simulators, and dataset loaders for the
Machine Learning for Human Learners tutorial."""

__version__ = "0.1.0"

from .data import (  # noqa: E402
    dass_items,
    load_dass,
    load_dollar_street,
    load_finch,
    load_household_survey,
    load_music_emotions,
    load_musiclab,
    load_perinatal_dev,
    load_things,
    load_wordbank,
)
from .plotting import COLORS, PALETTE, set_style  # noqa: E402
from .theming import enable_theme_plots  # noqa: E402
from .widgets import list_widgets, manifest, reset_assets, show, widget_html  # noqa: E402

__all__ = [
    "show", "widget_html", "list_widgets", "manifest", "reset_assets",
    "set_style", "enable_theme_plots", "COLORS", "PALETTE", "__version__",
    "load_dollar_street", "load_things", "load_finch", "load_dass", "dass_items",
    "load_wordbank", "load_perinatal_dev", "load_musiclab", "load_music_emotions", "load_household_survey",
]
