/* ══════════════════════════════════════════════════════════════════
   Figure 5 — Logistic regression, learning a boundary
   Gradient descent on the same objective scikit-learn's default
   LogisticRegression minimizes (log loss + L2, C = 1), so the walking
   solution approaches a Newton reference computed in JavaScript.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('logistic_gd', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Logistic regression, learning a boundary',
    hint: 'Press play &middot; scrub through training &middot; change the overlap'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 4, STEPS = cfg.steps || 300, CINV = 1;
  var X, y, names, dom, frames, sk;

  function simulate() {
    var rng = M.rng(seed), n = cfg.n || 70, ov = overlapS ? overlapS.get() : (cfg.overlap == null ? 0.45 : cfg.overlap);
    var a = rng.uniform(0, Math.PI * 2), sep = 1.25 * (1 - ov) + 0.1, X0 = [], y0 = [];
    for (var c = 0; c < 2; c++) {
      var s = c ? 1 : -1, cx = s * sep * Math.cos(a) / 2, cy = s * sep * Math.sin(a) / 2;
      for (var i = 0; i < n; i++) { X0.push([cx + rng.normal(0, 0.42), cy + rng.normal(0, 0.42)]); y0.push(c); }
    }
    X = X0; y = y0; names = ['category A', 'category B'];
    dom = { x: [-1.9, 1.9], y: [-1.5, 1.5] };
  }
  function loadReal() {
    X = cfg.data.X; y = cfg.data.y; names = cfg.data.names || ['class 0', 'class 1'];
    dom = M.bounds(X, 0.12);
  }

  function lossAt(w) {
    var L = 0, n = X.length;
    for (var i = 0; i < n; i++) {
      var z = w[0] * X[i][0] + w[1] * X[i][1] + w[2];
      L += y[i] ? (Math.max(0, -z) + Math.log1p(Math.exp(-Math.abs(z)))) : (Math.max(0, z) + Math.log1p(Math.exp(-Math.abs(z))));
    }
    return L / n + 0.5 * (w[0] * w[0] + w[1] * w[1]) / (CINV * n);
  }
  function accAt(w) {
    var c = 0;
    for (var i = 0; i < X.length; i++) if ((w[0] * X[i][0] + w[1] * X[i][1] + w[2] > 0 ? 1 : 0) === y[i]) c++;
    return c / X.length;
  }
  function grad(w) {
    var g = [0, 0, 0], n = X.length;
    for (var i = 0; i < n; i++) {
      var r = M.sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + w[2]) - y[i];
      g[0] += r * X[i][0]; g[1] += r * X[i][1]; g[2] += r;
    }
    return [g[0] / n + w[0] / (CINV * n), g[1] / n + w[1] / (CINV * n), g[2] / n];
  }
  function newton() {
    var w = [0, 0, 0], n = X.length;
    for (var it = 0; it < 30; it++) {
      var H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], g = grad(w);
      for (var i = 0; i < n; i++) {
        var p = M.sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + w[2]), q = p * (1 - p) / n, f = [X[i][0], X[i][1], 1];
        for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) H[a][b] += q * f[a] * f[b];
      }
      H[0][0] += 1 / (CINV * n); H[1][1] += 1 / (CINV * n);
      var d = M.solve(H, g);
      w = [w[0] - d[0], w[1] - d[1], w[2] - d[2]];
      if (Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]) < 1e-9) break;
    }
    return { w: w, loss: lossAt(w), acc: accAt(w) };
  }
  function train() {
    var lr = lrS.get(), w = [0, 0, 0];
    frames = [{ w: w.slice(), loss: lossAt(w), acc: accAt(w) }];
    for (var t = 0; t < STEPS; t++) {
      var g = grad(w);
      w = [w[0] - lr * g[0], w[1] - lr * g[1], w[2] - lr * g[2]];
      frames.push({ w: w.slice(), loss: lossAt(w), acc: accAt(w) });
    }
    tr.setN(frames.length, true);
  }
  function rebuild() { if (real) loadReal(); else simulate(); sk = newton(); train(); drawAll(); }

  /* ─── DOM ─── */
  var split = M.split(root, 1.7);
  M.mini(split.a, 'probability of ' + (real ? (cfg.data.names || ['', 'class 1'])[1] : 'category B'));
  M.mini(split.b, 'error (log loss) during training');
  var main = M.stage(split.a, { height: function (w) { return M.clamp(w * 0.62, 230, 380); }, ref: 520, label: 'Probability map from logistic regression with the decision boundary' });
  var side = M.stage(split.b, { height: function () { return main.h || 260; }, ref: 300, label: 'Training loss curve' });
  var legend = M.legend(root, []);
  var tr = M.transport(root, {
    fps: 30, n: 1,
    fmt: function (i, n) { return 'step ' + i + ' / ' + (n - 1); },
    stride: function (i) { return i < 40 ? 1 : i < 120 ? 2 : 4; },
    onFrame: function () { drawAll(); }
  });
  var row = M.row(root);
  var lrS = M.slider(row, { label: 'Step size', min: 0.05, max: 4, value: cfg.lr || 0.6, log: true, accent: 'green', fmt: function (v) { return v.toFixed(2); }, onInput: function () { train(); drawAll(); } });
  var overlapS = null;
  if (!real) overlapS = M.slider(row, { label: 'Overlap', min: 0, max: 1, step: 0.02, value: cfg.overlap == null ? 0.45 : cfg.overlap, accent: 'orange', fmt: function (v) { return v.toFixed(2); }, onInput: rebuild });
  var row2 = M.row(root);
  var ghost = true;
  var ghostBtn = M.button(row2, 'Newton reference', function () { ghost = !ghost; ghostBtn.classList.toggle('on', ghost); drawAll(); }, { aria: 'Toggle the numerical Newton reference' });
  ghostBtn.classList.add('on');
  M.spacer(row2);
  if (!real) M.button(row2, 'new sample', function () { seed++; rebuild(); tr.show(0); }, { aria: 'Draw a new random sample' });
  M.resetButton(row2, function () { tr.pause(); lrS.set(cfg.lr || 0.6, true); if (overlapS) overlapS.set(cfg.overlap == null ? 0.45 : cfg.overlap, true); seed = cfg.seed || 4; rebuild(); tr.show(0); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var v = M.view(main, { x: [0, 1], y: [0, 1], pad: real ? { l: 50, b: 40 } : { l: 14, b: 14, t: 12, r: 12 } });
  var vs = M.view(side, { x: [0, 1], y: [0, 1], pad: { l: 40, b: 34, t: 12, r: 10 } });

  function boundary(ctx, w, p, color, lw, dash) {
    /* Line where w·x + b = logit(p), clipped to the view. */
    var t = Math.log(p / (1 - p)), pts = [];
    if (Math.abs(w[1]) > 1e-9) {
      [v.x0, v.x1].forEach(function (x) { pts.push([x, (t - w[2] - w[0] * x) / w[1]]); });
    } else if (Math.abs(w[0]) > 1e-9) {
      var x = (t - w[2]) / w[0]; pts.push([x, v.y0], [x, v.y1]);
    } else return;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(v.px(pts[0][0]), v.py(pts[0][1])); ctx.lineTo(v.px(pts[1][0]), v.py(pts[1][1])); ctx.stroke();
    ctx.restore();
  }

  function drawMain(f) {
    var ctx = main.ctx, s = main;
    if (!s.w) return;
    v.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    var w = f.w;
    M.field(ctx, v, 80, 60, function (x, yy) {
      var p = M.sigmoid(w[0] * x + w[1] * yy + w[2]), c = M.diverge(p);
      return [c[0], c[1], c[2], 150];
    });
    if (real) M.axes(ctx, v, s, { grid: false, xlabel: cfg.data.xlabel || 'feature 1', ylabel: cfg.data.ylabel || 'feature 2' });
    ctx.save(); v.clip(ctx);
    if (ghost) boundary(ctx, sk.w, 0.5, C.green, 2 * s.sc, [6 * s.sc, 4 * s.sc]);
    boundary(ctx, w, 0.25, M.rgba(C.ink, 0.45), 1 * s.sc, [3 * s.sc, 3 * s.sc]);
    boundary(ctx, w, 0.75, M.rgba(C.ink, 0.45), 1 * s.sc, [3 * s.sc, 3 * s.sc]);
    boundary(ctx, w, 0.5, C.ink, 2.2 * s.sc);
    for (var i = 0; i < X.length; i++) {
      var col = y[i] ? C.orange : C.blue;
      M.dot(ctx, v.px(X[i][0]), v.py(X[i][1]), 3.5 * s.sc, col, '#fff', 1);
    }
    ctx.restore();
  }

  function drawSide(i) {
    var ctx = side.ctx, s = side;
    if (!s.w) return;
    var hi = frames[0].loss * 1.05, lo = Math.min(sk.loss, frames[frames.length - 1].loss);
    vs.setDomain([0, frames.length - 1], [Math.max(0, lo - (hi - lo) * 0.08), hi]);
    M.clear(ctx, s);
    M.axes(ctx, vs, s, { xlabel: 'step', ylabel: 'error', nx: 3, ny: 4 });
    ctx.save(); vs.clip(ctx);
    ctx.setLineDash([5 * s.sc, 4 * s.sc]); ctx.strokeStyle = C.green; ctx.lineWidth = 1.4 * s.sc;
    ctx.beginPath(); ctx.moveTo(vs.L, vs.py(sk.loss)); ctx.lineTo(vs.R, vs.py(sk.loss)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = M.rgba(C.blue, 0.25); ctx.lineWidth = 1.5 * s.sc;
    ctx.beginPath();
    frames.forEach(function (f, k) { ctx[k ? 'lineTo' : 'moveTo'](vs.px(k), vs.py(f.loss)); });
    ctx.stroke();
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2 * s.sc;
    ctx.beginPath();
    for (var k = 0; k <= i; k++) ctx[k ? 'lineTo' : 'moveTo'](vs.px(k), vs.py(frames[k].loss));
    ctx.stroke();
    M.dot(ctx, vs.px(i), vs.py(frames[i].loss), 4 * s.sc, C.blue, '#fff', 1.5);
    ctx.restore();
    ctx.font = M.font(s, 8.5, 500); ctx.fillStyle = C.green; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText('SKLEARN', vs.R - 4 * s.sc, vs.py(sk.loss) - 3 * s.sc);
  }

  function drawAll() {
    if (!frames) return;
    var i = tr.i, f = frames[i];
    drawMain(f); drawSide(i);
    M.setLegend(legend, [
      { color: C.blue, label: names[0] }, { color: C.orange, label: names[1] },
      { color: C.ink, label: 'boundary (50 / 50)', line: true }, { color: C.green, label: 'Newton reference (same objective)', line: true }
    ]);
    var gap = (f.loss - sk.loss) / Math.max(1e-9, sk.loss);
    var diverged = !isFinite(f.loss) || f.loss > frames[0].loss * 1.5;
    var status = diverged ? '<span class="r">the step size is too big: each step overshoots</span>'
      : gap < 0.005 ? '<span class="g">matches scikit-learn\'s answer</span>'
      : i === 0 ? '<span class="k">every point starts at 50 / 50: the model knows nothing yet</span>'
      : '<span class="b">sharpening the boundary&hellip;</span>';
    read.innerHTML = '<span class="k">step</span> ' + i + '<span class="sep">&middot;</span><span class="k">error</span> ' + M.f(f.loss, 3) +
      '<span class="sep">&middot;</span><span class="k">accuracy</span> ' + Math.round(f.acc * 100) + '%' +
      '<span class="sep">&middot;</span>' + status;
  }

  rebuild();
  main.draw = drawAll; side.draw = drawAll;
  main.onResize = function () { setTimeout(side.resize, 0); };
  main.request();
  return { destroy: function () { tr.pause(); } };
});
