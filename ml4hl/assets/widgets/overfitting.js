/* ══════════════════════════════════════════════════════════════════
   Figure 14 — Underfitting, overfitting, and the sweet spot
   Fit polynomials of increasing flexibility. Training error only ever
   goes down; error on fresh data goes down, then back up.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('overfitting', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Underfitting, overfitting, and the sweet spot',
    hint: 'Slide the flexibility (polynomial degree) &middot; resample the data &middot; watch the test error'
  });

  var real = cfg.data && cfg.data.x;
  var seed = cfg.seed || 14, MAXD = 15;
  var tr, te, xr, yr, fits, errs, truth = null;

  function f(x) { return 0.65 * Math.sin(3.1 * x) + 0.35 * x; }
  function load() {
    var rng = M.rng(seed);
    if (real) {
      var xs = cfg.data.x, ys = cfg.data.y, idx = rng.shuffle(M.range(xs.length)), half = Math.max(4, Math.round(xs.length * 0.5));
      var lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs);
      xr = [lo, hi]; yr = [Math.min.apply(null, ys), Math.max.apply(null, ys)];
      var sx = function (x) { return 2 * (x - lo) / ((hi - lo) || 1) - 1; };
      tr = idx.slice(0, half).map(function (i) { return [sx(xs[i]), ys[i]]; });
      te = idx.slice(half).map(function (i) { return [sx(xs[i]), ys[i]]; });
    } else {
      var n = nS.get(), sd = noiseS.get();
      xr = [-1, 1];
      tr = M.range(n).map(function () { var x = rng.uniform(-1, 1); return [x, f(x) + rng.normal(0, sd)]; });
      te = M.range(80).map(function () { var x = rng.uniform(-1, 1); return [x, f(x) + rng.normal(0, sd)]; });
      truth = f;
      yr = [-1.4, 1.4];
    }
    fitAll();
  }

  function basis(x, d) {
    var p = [1];
    if (d >= 1) p.push(x);
    for (var k = 1; k < d; k++) p.push(((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1));
    return p;
  }
  function fitAll() {
    var ys = tr.map(function (p) { return p[1]; }), my = M.mean(ys), sy = M.std(ys) || 1;
    fits = []; errs = [];
    for (var d = 0; d <= MAXD; d++) {
      var Phi = tr.map(function (p) { return basis(p[0], d); });
      var w = M.lstsq(Phi, ys.map(function (y) { return (y - my) / sy; }), d + 1 >= tr.length ? 1e-6 : 1e-9);
      var pred = (function (w) { return function (x) { var b = basis(x, d), s = 0; for (var i = 0; i < b.length; i++) s += w[i] * b[i]; return my + sy * s; }; })(w);
      fits.push(pred);
      var mse = function (set) { return M.mean(set.map(function (p) { var e = pred(p[0]) - p[1]; return e * e; })); };
      errs.push([mse(tr), mse(te)]);
    }
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.3);
  M.mini(split.a, 'the fit');
  M.mini(split.b, 'error vs flexibility');
  var hf = function (w) { return M.clamp(w * 0.72, 230, 340); };
  var fit = M.stage(split.a, { height: hf, ref: 460, label: 'Training and test points with a polynomial fit' });
  var errSt = M.stage(split.b, { height: hf, ref: 360, label: 'Training and test error against polynomial degree' });
  var legend = M.legend(root, [
    { color: C.blue, label: 'training points (the model sees these)' },
    { color: C.orange, label: 'test points (held out)', ring: true },
    { color: C.ink, label: 'model', line: true }
  ].concat(real ? [] : [{ color: C.green, label: 'the true pattern', line: true }]));
  legend.style.margin = '6px 0 0';
  var row = M.row(root);
  var dS = M.slider(row, { label: 'Flexibility (degree)', min: 0, max: MAXD, step: 1, value: cfg.degree == null ? 3 : cfg.degree, accent: 'blue', onInput: function () { draw(); } });
  var row2 = M.row(root);
  var nS = real ? null : M.slider(row2, { label: 'Training points', min: 8, max: 80, step: 1, value: cfg.n || 18, accent: 'blue', onChange: function () { load(); draw(); } });
  var noiseS = real ? null : M.slider(row2, { label: 'Noise', min: 0.05, max: 1, step: 0.05, value: cfg.noise || 0.3, accent: 'orange', onChange: function () { load(); draw(); } });
  var row3 = M.row(root);
  M.button(row3, real ? 'new split' : 'new sample', function () { seed++; load(); draw(); }, { aria: 'Draw a new sample or train/test split' });
  M.spacer(row3);
  M.resetButton(row3, function () {
    seed = cfg.seed || 14; dS.set(cfg.degree == null ? 3 : cfg.degree, true);
    if (nS) nS.set(cfg.n || 18, true); if (noiseS) noiseS.set(cfg.noise || 0.3, true);
    load(); draw();
  });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var vF = M.view(fit, { x: [-1, 1], y: [-1, 1], pad: real ? { l: 52, r: 12, t: 12, b: 38 } : { l: 14, r: 12, t: 12, b: 14 } });
  function drawFit() {
    var ctx = fit.ctx, s = fit, d = dS.get();
    if (!s.w) return;
    var py = (yr[1] - yr[0]) * 0.12;
    vF.setDomain([-1.04, 1.04], [yr[0] - py, yr[1] + py]);
    M.clear(ctx, s);
    if (real) M.axes(ctx, vF, s, { xlabel: cfg.data.xlabel || 'x', ylabel: cfg.data.ylabel || 'y', xlabels: false, ny: 4 });
    ctx.save(); vF.clip(ctx);
    if (truth) {
      ctx.strokeStyle = M.rgba(C.green, 0.8); ctx.lineWidth = 1.6 * s.sc; ctx.setLineDash([6 * s.sc, 4 * s.sc]);
      ctx.beginPath();
      for (var i = 0; i <= 200; i++) { var x = -1.04 + 2.08 * i / 200; ctx[i ? 'lineTo' : 'moveTo'](vF.px(x), vF.py(truth(x))); }
      ctx.stroke(); ctx.setLineDash([]);
    }
    te.forEach(function (p) { M.dot(ctx, vF.px(p[0]), vF.py(p[1]), 3.4 * s.sc, null, M.rgba(C.orange, 0.85), 1.4 * s.sc); });
    tr.forEach(function (p) { M.dot(ctx, vF.px(p[0]), vF.py(p[1]), 4 * s.sc, C.blue, '#fff', 1); });
    var span = yr[1] - yr[0], ylo = yr[0] - 2 * span, yhi = yr[1] + 2 * span;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.4 * s.sc; ctx.beginPath();
    for (i = 0; i <= 400; i++) { var xx = -1.04 + 2.08 * i / 400; ctx[i ? 'lineTo' : 'moveTo'](vF.px(xx), vF.py(M.clamp(fits[d](xx), ylo, yhi))); }
    ctx.stroke();
    ctx.restore();
  }

  var vE = M.view(errSt, { x: [0, MAXD], y: [0, 1], pad: { l: 46, r: 12, t: 12, b: 38 } });
  function drawErr() {
    var ctx = errSt.ctx, s = errSt, d = dS.get();
    if (!s.w) return;
    var L = function (v) { return Math.log10(Math.max(v, 1e-6)); };
    var floor = L(Math.min.apply(null, errs.map(function (e) { return e[0]; }))), cap = L(errs[0][1]) + 1.2;
    var lo = Math.max(floor, cap - 4.5) - 0.1, hi = cap;
    vE.setDomain([-0.4, MAXD + 0.4], [lo, hi]);
    M.clear(ctx, s);
    M.axes(ctx, vE, s, { xlabel: 'flexibility (degree)', ylabel: 'error (log)', xticks: [0, 3, 6, 9, 12, 15], ylabels: false, ny: 4 });
    ctx.save(); vE.clip(ctx);
    var best = 0;
    errs.forEach(function (e, i) { if (e[1] < errs[best][1]) best = i; });
    ctx.fillStyle = M.rgba(C.green, 0.1);
    ctx.fillRect(vE.px(best - 0.5), vE.T, vE.px(best + 0.5) - vE.px(best - 0.5), vE.B - vE.T);
    [[0, C.blue], [1, C.orange]].forEach(function (k) {
      ctx.strokeStyle = k[1]; ctx.lineWidth = 2 * s.sc; ctx.beginPath();
      errs.forEach(function (e, i) { ctx[i ? 'lineTo' : 'moveTo'](vE.px(i), vE.py(Math.min(hi, L(e[k[0]])))); });
      ctx.stroke();
      errs.forEach(function (e, i) { M.dot(ctx, vE.px(i), vE.py(Math.min(hi, L(e[k[0]]))), 2.2 * s.sc, k[1], null); });
    });
    ctx.strokeStyle = M.rgba(C.ink, 0.5); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(vE.px(d) + 0.5, vE.T); ctx.lineTo(vE.px(d) + 0.5, vE.B); ctx.stroke(); ctx.setLineDash([]);
    M.dot(ctx, vE.px(d), vE.py(Math.min(hi, L(errs[d][0]))), 5 * s.sc, C.blue, '#fff', 1.5 * s.sc);
    M.dot(ctx, vE.px(d), vE.py(Math.min(hi, L(errs[d][1]))), 5 * s.sc, C.orange, '#fff', 1.5 * s.sc);
    ctx.restore();
    M.pill(ctx, s, 'sweet spot', vE.px(best), vE.T + 10 * s.sc, { align: 'center', color: C.green, size: 9 });
    ctx.font = M.font(s, 9, 500); ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillStyle = C.orange; ctx.fillText('test', vE.R - 4 * s.sc, vE.T + 26 * s.sc);
    ctx.fillStyle = C.blue; ctx.fillText('train', vE.R - 4 * s.sc, vE.T + 38 * s.sc);
    return best;
  }

  function draw() {
    if (!fits) return;
    drawFit();
    var best = drawErr(), d = dS.get(), e = errs[d];
    var verdict = d < best - 1 ? '<span class="o">underfitting: too stiff to follow the pattern</span>'
      : d > best + 2 && e[1] > errs[best][1] * 1.15 ? '<span class="r">overfitting: memorizing noise, worse on new data</span>'
      : '<span class="g">near the sweet spot</span>';
    read.innerHTML = '<span class="k">degree</span> ' + d +
      '<span class="sep">&middot;</span><span class="b">train error</span> ' + M.f(e[0], e[0] > 100 ? 0 : 3) +
      '<span class="sep">&middot;</span><span class="o">test error</span> ' + M.f(e[1], e[1] > 100 ? 0 : 3) +
      '<span class="sep">&middot;</span>' + verdict;
  }

  load();
  fit.draw = draw; errSt.draw = draw;
  fit.request();
});
