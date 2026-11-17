# GitHub Automation Index

This directory defines continuous validation, Google Colab runtime verification, and GitHub Pages deployment for `ML4HL-Workshop`.

## Contents

- `INDEX.md` — Overview of `.github/` workflows, jobs, pinned actions, and deployment settings.
- `workflows/pages.yml` — **Workshop Pages** workflow triggered on pushes to `main`, pull requests, and `workflow_dispatch`.

## Workflow Jobs (`workflows/pages.yml`)

1. **`build`** (`ubuntu-24.04`): Installs Python 3.12, Node 22, Quarto 1.5.57, and `ml4hl[tutorial]`, regenerates theme assets and `ML4HL-Tutorial.ipynb`, runs `check_data_release.py` and `check_science.py`, renders `ML4HL-Tutorial.qmd` with `--no-execute-daemon`, checks outputs (`check_outputs.py`), packages `_site` (`index.html`, `ML4HL-Tutorial.ipynb`, `ML4HL-Tutorial.qmd`, `ML4HL-Source.zip`, licenses, `.nojekyll`) via `build_workshop.py`, validates links and exclusions (`validate_site.py`), and uploads the `site` artifact.
2. **`colab`** (`ubuntu-24.04`): Reclaims ephemeral runner disk space, installs `Colab-Test`, executes `ML4HL-Tutorial.ipynb` end-to-end inside the pinned official Google Colab runtime image (`check_colab.py`), and uploads `colab-runtime-evidence`.
3. **`deploy`** (`ubuntu-24.04`, `needs: [build, colab]`): On `main` pushes when repository variable `PAGES_DEPLOY_ENABLED` is `'true'`, downloads the validated `site` artifact and publishes `./_site` to the `gh-pages` branch (`force_orphan: false`), serving the site at `https://colinconwell.github.io/ML4HL-Workshop/`.

## Actions Used

- `actions/checkout@v4` — Repository checkout for `build` and `colab`.
- `actions/setup-python@v5` — Python 3.12 toolchain setup.
- `actions/setup-node@v4` — Node 22 runtime for JavaScript widget math verification (`check_science.py`).
- `actions/upload-artifact@v4` — Uploads the packaged `_site` bundle and Colab runtime logs.
- `actions/download-artifact@v4` — Retrieves the `site` artifact in `deploy`.
- `peaceiris/actions-gh-pages@329bcc8f12caed2cefe5a5b80781499a6f3b361b` (`v4`) — Commits and pushes `./_site` to the `gh-pages` branch using `GITHUB_TOKEN`.
