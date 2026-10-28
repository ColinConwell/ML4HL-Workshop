#!/usr/bin/env bash
# Execute the Colab notebook natively (arm64) with the Colab runtime's exact package versions.
#
# On Apple Silicon, `colab-test` runs the amd64 Colab image under emulation, where Jupyter
# kernels deadlock as soon as anything spawns a subprocess (seaborn's import does). This
# script reads the version pins from the Colab image instead and runs the notebook in a
# native python:3.12-slim container, which checks API compatibility with Colab's libraries.
#
# Usage: scripts/colab_pinned.sh [notebook.ipynb]
set -euo pipefail

NB="${1:-ML4HL-Tutorial.ipynb}"
IMAGE="${COLAB_IMAGE:-us-docker.pkg.dev/colab-images/public/runtime:latest}"
OUT="${2:-.cache/colab-pinned}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKGS='numpy|pandas|scipy|scikit-learn|scikit-image|matplotlib|matplotlib-inline|seaborn|huggingface-hub|hf-xet|fsspec|pyarrow|ipython|ipykernel|jupyter_client|nbformat|nbconvert|joblib|threadpoolctl|networkx|h5py|pillow|requests|tqdm|torch|torchvision'

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$NB" "$WORK/notebook.ipynb"
cp "$SCRIPT_DIR/execute_notebook.py" "$WORK/execute_notebook.py"
docker image inspect "$IMAGE" > "$OUT/runtime-image.json"

echo "Reading package pins from the Colab image..."
docker run --rm --platform linux/amd64 --entrypoint /bin/bash "$IMAGE" -c 'pip freeze 2>/dev/null' \
  | grep -iE "^(${PKGS})==" | sed -E 's/\+cu[0-9]+//' > "$WORK/pins.txt"
grep -iE '^(scikit-learn|pandas|ipython)==' "$WORK/pins.txt" | tr '\n' ' '; echo

echo "Executing $NB in python:3.12-slim (arm64) with those pins..."
cp "$WORK/pins.txt" "$OUT/pins.txt"
docker run --rm --platform linux/arm64 -e JOBLIB_MULTIPROCESSING=0 -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 -e ML4HL_RUNTIME_IMAGE="native-arm64-with-pins-from-$IMAGE" -w /nb -v "$WORK:/nb:rw" -v "$OUT:/results:rw" python:3.12-slim bash -c '
apt-get update -qq >/dev/null && apt-get install -y -qq gcc g++ >/dev/null 2>&1
pip install -q uv >/dev/null 2>&1
uv pip install --system -q -r pins.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
python execute_notebook.py notebook.ipynb --output /results/executed.ipynb'
