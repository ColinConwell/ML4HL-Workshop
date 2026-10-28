# Machine Learning for Human Learners

An executable workshop by Colin Conwell, VISNA Lab. The Quarto source introduces regression, classification, clustering, dimensionality reduction, representational similarity, and evaluation through interactive figures and public research datasets.

[Read the Executed Tutorial](https://colinconwell.github.io/ML4HL-Workshop/) · [Run in Colab](https://colab.research.google.com/github/ColinConwell/ML4HL-Workshop/blob/main/ML4HL-Tutorial.ipynb)

## Build and Validate

Use Python 3.12, Quarto 1.5.57 or later, and Node.js 20 or later. Run `python -m pip install '.[tutorial]'`, then `python scripts/build_theme.py`, `python scripts/to_colab.py`, and `quarto render ML4HL-Tutorial.qmd --no-execute-daemon`. Run `python scripts/check_science.py` for the numerical reference checks. `ML4HL-Tutorial.qmd` is the source of truth; regenerate the notebook instead of editing it directly.

The notebook carries a checksum-verified copy of the reviewed helper code. Dataset artifacts are downloaded from the immutable Hugging Face revision in `ml4hl/assets/data-release.json`. Its setup installs tutorial packages missing from Colab. The VISNA logo remains on its original white background. Page appearance and plot rendering can be controlled independently.

## Execution and Publication

The Pages workflow renders with execution enabled, checks numerical routines and local links, runs Colab-Test on a native x86 runner, and publishes to `gh-pages` only after all required checks succeed. The official Colab image can freeze under Apple Silicon emulation. `scripts/colab_pinned.sh` is a separate native ARM fallback using that image's package versions; passing it does not establish official-runtime execution.

## Source and Licenses

This repository is a managed export from Colin Conwell's Tutorials collection. `.ml4hl-export.json` records the exported files so future synchronizations can preserve unrelated additions and detect remote edits. The source collection's GPL license is retained in `LICENSE`; the helper package's existing MIT declaration is retained with `LICENSE-HELPER`. Dataset licenses and attribution remain with their upstream sources. See `assets/README.md` for logo provenance. No license is claimed for third-party artwork or datasets beyond their existing terms.
