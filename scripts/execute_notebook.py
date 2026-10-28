"""Execute an isolated notebook and retain outputs and runtime provenance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path, default=Path('executed.ipynb'))
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, 4)
    start = time.time()
    receipt = dict(source_sha256=hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
                   platform=platform.platform(), python=platform.python_version(),
                   runtime_image=os.environ.get('ML4HL_RUNTIME_IMAGE', 'local'),
                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        ExecutePreprocessor(timeout=900, kernel_name='python3').preprocess(
            notebook, {'metadata': {'path': str(args.notebook.absolute().parent)}})
        cells = [c for c in notebook.cells if c.cell_type == 'code' and c.source.strip()]
        assert all(c.execution_count is not None for c in cells), 'Unexecuted code cell'
        assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
        assert sum(bool(c.outputs) for c in cells) >= 45, 'Required outputs missing'
        receipt.update(status='passed', code_cells=len(cells), output_cells=sum(bool(c.outputs) for c in cells))
    except Exception as error:
        receipt.update(status='failed', error=str(error)[-5000:])
        raise
    finally:
        receipt['elapsed_seconds'] = round(time.time()-start, 2)
        receipt['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
        args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k!='packages'}, indent=2))

if __name__ == '__main__': main()
