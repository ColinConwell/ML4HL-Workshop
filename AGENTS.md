# ML4HL Workshop

`ML4HL-Tutorial.qmd` is the source of truth. `ml4hl/` contains Python loaders, plotting helpers, and vanilla JavaScript widgets. `assets/` contains the unchanged VISNA logo and generated theme header. `scripts/` contains build, numerical validation, notebook conversion, runtime checks, and site packaging. Generated HTML includes executed outputs; generated notebooks bundle the matching helper source.

Use Python 3.12 and Quarto with `--no-execute-daemon`. Widgets emit assets once per kernel, but once per output in Colab. Do not resolve Hub snapshot symlinks when locating package assets. Title Case applies to heading fragments. Keep Markdown paragraphs on single source lines. Publication must pass official Colab-runtime validation on native x86; the native ARM fallback is a separate result. Never publish credentials, local caches, motif slides, plans, or the local widget gallery.
