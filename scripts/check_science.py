"""Compare shared JavaScript numerical routines with independent Python references."""
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
ROOT = Path(__file__).absolute().parents[1]
rng = np.random.default_rng(402)
cases = [rng.normal(size=(37, 5)), rng.normal(size=(8, 19)), np.ones((4, 3))]
x = [1, 2, 4, 4, 9, 12]
y = [2, -1, 3, 3, 6, 10]
script = '''const fs=require('fs'),vm=require('vm');
const sandbox={window:{},document:{},console,setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
const M=sandbox.window.ML4HL, d=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify({fit:M.linfit(d.x,d.y),rank:M.rank(d.x),rho:M.spearman(d.x,d.y),pca:d.cases.map(x=>M.pca(x,2))}));'''
raw = subprocess.check_output(['node','-e',script,str(ROOT/'ml4hl/assets/core.js')], input=json.dumps(dict(x=x,y=y,cases=[a.tolist() for a in cases])).encode())
actual=json.loads(raw)
fit=LinearRegression().fit(np.array(x)[:,None],y)
np.testing.assert_allclose([actual['fit']['m'],actual['fit']['b']],[fit.coef_[0],fit.intercept_])
np.testing.assert_allclose(actual['rank'],rankdata(x))
np.testing.assert_allclose(actual['rho'],spearmanr(x,y).statistic)
for X, result in zip(cases,actual['pca']):
 if np.ptp(X)==0:
  np.testing.assert_allclose(result['values'],0,atol=1e-12)
  continue
 reference=PCA(2).fit(X)
 np.testing.assert_allclose(result['values'][:2],reference.explained_variance_,atol=1e-8)
 A=np.array(result['components']);B=reference.components_
 np.testing.assert_allclose(A.T@A,B.T@B,atol=1e-8)
# Evaluate the widget's own Newton solver and gradient against sklearn/finite differences.
X=rng.normal(size=(90,2));labels=(X[:,0]-.7*X[:,1]+rng.normal(size=90)>.1).astype(int)
widget=(ROOT/'ml4hl/assets/widgets/logistic_gd.js').read_text()
prefix=widget.split('  /* ─── DOM')[0]
solver=prefix+"loadReal(); return {solution:newton(), gradient:grad([.2,-.3,.1]), losses:[lossAt([.200001,-.3,.1]),lossAt([.199999,-.3,.1])]}; });"
js=script[:script.index('console.log')]+"M.frame=()=>{};sandbox.ML4HL=M;M.register=(name,fn)=>{sandbox.answer=fn({}, {data:{X:d.X,y:d.y}})};vm.runInNewContext(d.solver,sandbox);console.log(JSON.stringify(sandbox.answer));"
r=json.loads(subprocess.check_output(['node','-e',js,str(ROOT/'ml4hl/assets/core.js')],input=json.dumps(dict(X=X.tolist(),y=labels.tolist(),solver=solver)).encode()))
from sklearn.linear_model import LogisticRegression
reference=LogisticRegression(C=1,tol=1e-12,max_iter=2000).fit(X,labels)
np.testing.assert_allclose(r['solution']['w'],np.r_[reference.coef_[0],reference.intercept_],atol=1e-6)
np.testing.assert_allclose(r['gradient'][0],(r['losses'][0]-r['losses'][1])/2e-6,atol=1e-8)
# Ensure rendering variants does not change computed artists or image pixels.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(ROOT))
from ml4hl.plotting import set_style
from ml4hl.theming import figure_html, THEME
set_style();fig,ax=plt.subplots();line,=ax.plot(x,y);im=ax.imshow(np.arange(4).reshape(2,2),cmap='viridis')
color=line.get_color();pixels=im.get_array().copy();html=figure_html(fig)
assert html.count('<img ')==3 and line.get_color()==color
np.testing.assert_array_equal(im.get_array(),pixels)
for mode in ('light','dark'):
 assert len(set(THEME[mode][k] for k in THEME['categorical']))==12
print('PASS: OLS, tied ranks, Spearman, covariance/Gram PCA, constant-data PCA, plot preservation, 12 distinct category colors, logistic Newton/gradient references')
