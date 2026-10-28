"""Run Colab-Test preflight and unmodified-setup execution in its official image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat
ROOT=Path(__file__).absolute().parents[1]
IMAGE=json.loads((ROOT/'ml4hl/assets/runtime.json').read_text())['official_colab_image']
COLAB_TEST_REVISION='ee5bcec8280c4e6261ccb03b15ee18a9a79aaaa0'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'.cache/colab-official');a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 print(f'Pulling pinned official image: {IMAGE}',flush=True)
 subprocess.run(['docker','pull','--platform','linux/amd64',IMAGE],check=True,timeout=1800)
 # Colab-Test uses this fixed tag internally; bind it to the recorded immutable image.
 subprocess.run(['docker','tag',IMAGE,'us-docker.pkg.dev/colab-images/public/runtime:latest'],check=True)
 (a.output/'runtime-image.json').write_text(subprocess.check_output(['docker','image','inspect',IMAGE],text=True))
 env=dict(os.environ)
 if a.project:env['PYTHONPATH']=str(a.project.absolute())
 source=ROOT/'ML4HL-Tutorial.ipynb';nb=nbformat.read(source,4)
 # The final evidence cell runs only after all tutorial cells succeed; setup stays unchanged.
 nb.cells.append(nbformat.v4.new_code_cell('''import importlib.metadata, json, platform
from pathlib import Path
from threadpoolctl import threadpool_info
Path("runtime-result.json").write_text(json.dumps({"status":"passed","completed_tutorial_code_cells":get_ipython().execution_count-1,"threadpools":threadpool_info(),"python":platform.python_version(),"platform":platform.platform(),"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},"dataset_release":ml4hl.data._RELEASE},indent=2))'''.replace('ml4hl.data._RELEASE', 'json.loads((Path(ml4hl.__file__).absolute().parent/"assets/data-release.json").read_text())')))
 target=a.output/'notebook.ipynb';nbformat.write(nb,target)
 (a.output/'source.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),colab_test_revision=COLAB_TEST_REVISION),indent=2))
 command=[sys.executable,'-m','colab_test.cli',str(target),'--platform','linux/amd64']
 print('Running package preflight; missing tutorial packages are expected before setup.',flush=True)
 with (a.output/'preflight.log').open('w') as log:
  preflight=subprocess.run(command+['--check-packages'],env=env,stdout=log,stderr=subprocess.STDOUT)
 # Preflight cannot infer pip installs or the bundled helper. Retain its result as evidence.
 (a.output/'preflight-status.json').write_text(json.dumps({'returncode':preflight.returncode,'note':'Missing tutorial packages are installed by the notebook setup; no auto-setup or preinstallation is used.'}))
 print('Executing all notebook cells in the pinned official runtime.',flush=True)
 with (a.output/'execution.log').open('w') as log:
  execution=subprocess.run(command+['--env',str(ROOT/'scripts/colab-test.env'),'--docker-logs'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
 if execution.returncode:
  print((a.output/'execution.log').read_text()[-15000:],file=sys.stderr)
 execution.check_returncode()
 assert (a.output/'runtime-result.json').is_file(), 'Runtime evidence cell did not finish'
 (a.output/'runtime-image.json').write_text(subprocess.check_output(['docker','image','inspect',IMAGE],text=True))
 print('PASS: official Colab runtime, notebook setup, all cells, and runtime receipt')
if __name__=='__main__':main()
