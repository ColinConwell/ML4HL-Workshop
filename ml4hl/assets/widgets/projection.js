/* ══════════════════════════════════════════════════════════════════
   Figure 12 — Projections: PCA and random
   PCA tab: rotate a line through the cloud; the shadow of every point on
   the line keeps some of the spread. The best line is the first principal
   component. Random tab: squash many dimensions into a few with a random
   matrix and watch how well pairwise distances survive.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('projection', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Projections: PCA and random',
    hint: 'Drag the line to rotate it (it snaps to the best one) &middot; switch to random projection'
  });

  var data = cfg.data || {};
  var seed = cfg.seed || 9, mode = cfg.mode || 'pca';

  /* ─── PCA tab state ─── */
  var P2, mu, angle = 0.25, pc1, varTot, curve;
  function loadPCA() {
    if (data.X2) P2 = data.X2.map(function (p) { return p.slice(); });
    else {
      var rng = M.rng(seed), r = corrS ? corrS.get() : (cfg.corr == null ? 0.8 : cfg.corr), n = 120;
      P2 = [];
      for (var i = 0; i < n; i++) {
        var a = rng.normal(0, 1), b = rng.normal(0, 1);
        P2.push([a, r * a + Math.sqrt(1 - r * r) * b]);
      }
    }
    mu = [M.mean(P2.map(function (p) { return p[0]; })), M.mean(P2.map(function (p) { return p[1]; }))];
    var sxx = 0, syy = 0, sxy = 0;
    P2.forEach(function (p) { var dx = p[0] - mu[0], dy = p[1] - mu[1]; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; });
    varTot = sxx + syy;
    pc1 = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    curve = [];
    for (var t = 0; t <= 90; t++) { var th = t / 90 * Math.PI; curve.push(captured(th)); }
  }
  function captured(th) {
    var c = Math.cos(th), s = Math.sin(th), v = 0;
    P2.forEach(function (p) { var d = (p[0] - mu[0]) * c + (p[1] - mu[1]) * s; v += d * d; });
    return v / varTot;
  }
  function wrapAngle(a) { a = a % Math.PI; return a < 0 ? a + Math.PI : a; }

  /* ─── Random tab state ─── */
  var XH, lab, dHigh, pairs, R, Y;
  function loadRandom() {
    if (data.Xhigh) { XH = data.Xhigh; lab = data.labels || null; }
    else {
      var rng = M.rng(seed + 50), d = dS ? dS.get() : (cfg.d || 200), g = 5, per = 30;
      var cents = [];
      for (var c = 0; c < g; c++) { var v = new Float64Array(d); for (var j = 0; j < d; j++) v[j] = rng.normal(0, 0.55); cents.push(v); }
      XH = []; lab = [];
      for (c = 0; c < g; c++) for (var i = 0; i < per; i++) {
        var x = new Float64Array(d);
        for (j = 0; j < d; j++) x[j] = cents[c][j] + rng.normal(0, 0.5);
        XH.push(x); lab.push(c);
      }
    }
    var prng = M.rng(seed + 77), n = XH.length;
    pairs = [];
    for (var p = 0; p < 700; p++) { var a = prng.int(n), b = prng.int(n); if (a !== b) pairs.push([a, b]); }
    dHigh = pairs.map(function (q) { return M.dist(XH[q[0]], XH[q[1]]); });
    newMatrix();
  }
  var rseed = 1;
  function newMatrix() {
    var d = XH[0].length, k = kS ? kS.get() : (cfg.k || 10), rng = M.rng(seed * 7 + rseed);
    R = [];
    for (var i = 0; i < k; i++) { var row = new Float64Array(d); for (var j = 0; j < d; j++) row[j] = rng.normal(0, 1 / Math.sqrt(k)); R.push(row); }
    Y = XH.map(function (x) { return R.map(function (r) { var s = 0; for (var j = 0; j < x.length; j++) s += r[j] * x[j]; return s; }); });
  }

  /* ─── DOM ─── */
  var modeRow = M.row(root);
  var modeT = M.toggle(modeRow, { label: 'Method', value: mode, options: [['pca', 'PCA (best line)'], ['random', 'random projection']], onChange: function (v) { mode = v; sync(); } });

  var pcaBox = M.h('div'); root.appendChild(pcaBox);
  var sp = M.split(pcaBox, 1.35);
  M.mini(sp.a, 'the data and a line through it');
  M.mini(sp.b, 'spread kept, for every angle of the line');
  var hf = function (w) { return M.clamp(w * 0.72, 230, 340); };
  var plane = M.stage(sp.a, { height: hf, ref: 460, label: 'Scatter plot with a rotatable projection line' });
  var curveSt = M.stage(sp.b, { height: hf, ref: 340, label: 'Curve of spread captured versus line angle' });
  var pRow = M.row(pcaBox);
  var corrS = data.X2 ? null : M.slider(pRow, { label: 'Correlation', min: 0, max: 0.97, step: 0.01, value: cfg.corr == null ? 0.8 : cfg.corr, accent: 'blue', onInput: function () { loadPCA(); draw(); } });
  M.spacer(pRow);
  M.button(pRow, 'snap to PC1', function () { tweenTo(pc1); }, { aria: 'Rotate the line to the first principal component' });

  var rndBox = M.h('div'); root.appendChild(rndBox);
  var sr = M.split(rndBox, 1);
  M.mini(sr.a, 'two of the new axes');
  M.mini(sr.b, 'distances before vs after squashing');
  var flat = M.stage(sr.a, { height: hf, ref: 400, label: 'Scatter plot of the data on two random axes' });
  var dist = M.stage(sr.b, { height: hf, ref: 400, label: 'Scatter of original versus projected pairwise distances' });
  var rRow = M.row(rndBox);
  var kS = M.slider(rRow, { label: 'Kept dimensions', min: 1, max: 100, step: 1, value: cfg.k || 10, accent: 'orange', onInput: function () { newMatrix(); draw(); } });
  var dS = data.Xhigh ? null : M.slider(rRow, { label: 'Original dimensions', min: 20, max: 400, step: 10, value: cfg.d || 200, accent: 'blue', onChange: function () { loadRandom(); draw(); } });
  var rRow2 = M.row(rndBox);
  M.button(rRow2, 'new random matrix', function () { rseed++; newMatrix(); draw(); }, { aria: 'Draw a new random projection matrix' });

  var row = M.row(root);
  M.spacer(row);
  if (!data.X2) M.button(row, 'new data', function () { seed++; loadPCA(); loadRandom(); draw(); }, { aria: 'Draw new random data' });
  M.resetButton(row, function () {
    seed = cfg.seed || 9; angle = 0.25; rseed = 1;
    if (corrS) corrS.set(cfg.corr == null ? 0.8 : cfg.corr, true);
    kS.set(cfg.k || 10, true); if (dS) dS.set(cfg.d || 200, true);
    loadPCA(); loadRandom(); draw();
  });
  var read = M.readout(root);

  function sync() {
    pcaBox.style.display = mode === 'pca' ? '' : 'none';
    rndBox.style.display = mode === 'random' ? '' : 'none';
    if (mode === 'random') { flat.resize(); dist.resize(); } else { plane.resize(); curveSt.resize(); }
    draw();
  }

  var tw = M.loop(function (dt) {
    var diff = wrapAngle(target - angle + Math.PI / 2) - Math.PI / 2;
    if (Math.abs(diff) < 0.003) { angle = target; draw(); return false; }
    angle = wrapAngle(angle + diff * Math.min(1, dt * 9));
    draw(); return true;
  });
  var target = 0;
  function tweenTo(a) { target = wrapAngle(a); tw.start(); }

  /* ─── Drawing: PCA ─── */
  var vP = M.view(plane, { x: [-1, 1], y: [-1, 1], equal: true, pad: { l: 12, r: 12, t: 12, b: 12 } });
  function drawPlane() {
    var ctx = plane.ctx, s = plane;
    if (!s.w) return;
    var bb = M.bounds(P2, 0.12);
    vP.setDomain(bb.x, bb.y);
    M.clear(ctx, s);
    var c = Math.cos(angle), sn = Math.sin(angle), L = 1e3;
    ctx.save(); vP.clip(ctx);
    // PC1 ghost
    var gc = Math.cos(pc1), gs = Math.sin(pc1);
    ctx.strokeStyle = M.rgba(C.green, 0.5); ctx.lineWidth = 1.2 * s.sc; ctx.setLineDash([5 * s.sc, 4 * s.sc]);
    ctx.beginPath(); ctx.moveTo(vP.px(mu[0] - L * gc), vP.py(mu[1] - L * gs)); ctx.lineTo(vP.px(mu[0] + L * gc), vP.py(mu[1] + L * gs)); ctx.stroke();
    ctx.setLineDash([]);
    // residual ticks and shadows
    ctx.lineWidth = 1;
    P2.forEach(function (p) {
      var d = (p[0] - mu[0]) * c + (p[1] - mu[1]) * sn, q = [mu[0] + d * c, mu[1] + d * sn];
      ctx.strokeStyle = M.rgba(C.ink, 0.12);
      ctx.beginPath(); ctx.moveTo(vP.px(p[0]), vP.py(p[1])); ctx.lineTo(vP.px(q[0]), vP.py(q[1])); ctx.stroke();
    });
    ctx.strokeStyle = C.orange; ctx.lineWidth = 2.4 * s.sc;
    ctx.beginPath(); ctx.moveTo(vP.px(mu[0] - L * c), vP.py(mu[1] - L * sn)); ctx.lineTo(vP.px(mu[0] + L * c), vP.py(mu[1] + L * sn)); ctx.stroke();
    P2.forEach(function (p) { M.dot(ctx, vP.px(p[0]), vP.py(p[1]), 3.2 * s.sc, M.rgba(C.blue, 0.85), '#fff', 0.7); });
    P2.forEach(function (p) {
      var d = (p[0] - mu[0]) * c + (p[1] - mu[1]) * sn;
      M.dot(ctx, vP.px(mu[0] + d * c), vP.py(mu[1] + d * sn), 2.4 * s.sc, C.orange, null);
    });
    // handle
    var hr = Math.min(bb.x[1] - bb.x[0], bb.y[1] - bb.y[0]) * 0.42;
    handle = [mu[0] + hr * c, mu[1] + hr * sn];
    M.dot(ctx, vP.px(handle[0]), vP.py(handle[1]), 7 * s.sc, '#fff', C.orange, 2 * s.sc);
    ctx.restore();
  }
  var handle = [0, 0];

  var vC = M.view(curveSt, { x: [0, 180], y: [0, 1], pad: { l: 44, r: 12, t: 12, b: 36 } });
  function drawCurve() {
    var ctx = curveSt.ctx, s = curveSt;
    if (!s.w) return;
    vC.setDomain([0, 180], [0, 1]);
    M.clear(ctx, s);
    M.axes(ctx, vC, s, { xlabel: 'angle of the line (degrees)', ylabel: 'spread kept', xticks: [0, 45, 90, 135, 180], yfmt: function (v) { return Math.round(v * 100) + '%'; }, ny: 4 });
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2 * s.sc; ctx.beginPath();
    curve.forEach(function (v, i) { ctx[i ? 'lineTo' : 'moveTo'](vC.px(i * 2), vC.py(v)); });
    ctx.stroke();
    var pd = wrapAngle(pc1) * 180 / Math.PI;
    ctx.strokeStyle = M.rgba(C.green, 0.6); ctx.setLineDash([4 * s.sc, 3 * s.sc]); ctx.lineWidth = 1.2 * s.sc;
    ctx.beginPath(); ctx.moveTo(vC.px(pd), vC.T); ctx.lineTo(vC.px(pd), vC.B); ctx.stroke(); ctx.setLineDash([]);
    M.pill(ctx, s, 'PC1', vC.px(pd), vC.T + 10 * s.sc, { align: 'center', color: C.green, size: 9 });
    var ad = wrapAngle(angle) * 180 / Math.PI;
    M.dot(ctx, vC.px(ad), vC.py(captured(angle)), 6 * s.sc, C.orange, '#fff', 2 * s.sc);
  }

  /* ─── Drawing: random ─── */
  var vF = M.view(flat, { x: [-1, 1], y: [-1, 1], equal: true, pad: { l: 12, r: 12, t: 12, b: 12 } });
  var vD = M.view(dist, { x: [0, 1], y: [0, 1], pad: { l: 48, r: 12, t: 12, b: 38 } });
  function drawFlat() {
    var ctx = flat.ctx, s = flat;
    if (!s.w) return;
    M.clear(ctx, s);
    var pts = Y.map(function (y) { return [y[0], y.length > 1 ? y[1] : 0]; });
    var bb = M.bounds(pts, 0.12);
    if (Y[0].length === 1) bb.y = [-1, 1];
    vF.setDomain(bb.x, bb.y);
    ctx.save(); vF.clip(ctx);
    pts.forEach(function (p, i) {
      var y = Y[0].length > 1 ? p[1] : ((i * 0.6180339) % 1) * 1.6 - 0.8;
      M.dot(ctx, vF.px(p[0]), vF.py(y), 3.4 * s.sc, lab ? M.catColor(lab[i]) : C.blue, '#fff', 0.7);
    });
    ctx.restore();
  }
  var corrNow = 0;
  function drawDist() {
    var ctx = dist.ctx, s = dist;
    if (!s.w) return;
    var dLow = pairs.map(function (q) { return M.dist(Y[q[0]], Y[q[1]]); });
    corrNow = M.pearson(dHigh, dLow);
    var hi = Math.max(Math.max.apply(null, dHigh), Math.max.apply(null, dLow)) * 1.05;
    vD.setDomain([0, hi], [0, hi]);
    M.clear(ctx, s);
    M.axes(ctx, vD, s, { xlabel: 'distance in all ' + XH[0].length + ' dims', ylabel: 'distance in ' + Y[0].length, ny: 4 });
    ctx.save(); vD.clip(ctx);
    ctx.strokeStyle = M.rgba(C.green, 0.7); ctx.setLineDash([5 * s.sc, 4 * s.sc]); ctx.lineWidth = 1.2 * s.sc;
    ctx.beginPath(); ctx.moveTo(vD.px(0), vD.py(0)); ctx.lineTo(vD.px(hi), vD.py(hi)); ctx.stroke(); ctx.setLineDash([]);
    for (var i = 0; i < pairs.length; i++) M.dot(ctx, vD.px(dHigh[i]), vD.py(dLow[i]), 2 * s.sc, M.rgba(C.orange, 0.45), null);
    ctx.restore();
    M.pill(ctx, s, 'agreement r = ' + M.f(corrNow, 2), vD.L + 6 * s.sc, vD.T + 12 * s.sc, { color: C.orange, size: 9.5 });
  }

  function draw() {
    if (mode === 'pca') {
      drawPlane(); drawCurve();
      var kept = captured(angle), best = captured(pc1), off = Math.abs(wrapAngle(angle - pc1 + Math.PI / 2) - Math.PI / 2) * 180 / Math.PI;
      read.innerHTML = '<span class="k">this line keeps</span> ' + M.pct(kept) + ' <span class="k">of the spread</span>' +
        '<span class="sep">&middot;</span><span class="k">best possible (PC1)</span> ' + M.pct(best) +
        '<span class="sep">&middot;</span>' + (off < 1.5 ? '<span class="g">you found PC1: the direction of maximum spread</span>' : '<span class="o">' + M.f(off, 0) + '&deg; away from the best line</span>');
    } else {
      drawFlat(); drawDist();
      var k = Y[0].length, d = XH[0].length;
      read.innerHTML = '<span class="k">' + d + ' numbers per point squashed to</span> ' + k +
        '<span class="sep">&middot;</span><span class="k">distance agreement</span> r = ' + M.f(corrNow, 2) +
        '<span class="sep">&middot;</span>' + (corrNow > 0.9 ? '<span class="g">distances survive almost intact</span>' : corrNow > 0.7 ? '<span class="b">rough shape kept</span>' : '<span class="o">too few dimensions: neighbors get scrambled</span>');
    }
  }

  var dragging = false;
  M.pointer(plane.cv, {
    down: function (p) { dragging = true; tw.stop(); setFrom(p); },
    drag: function (p) { if (dragging) setFrom(p); },
    up: function () { dragging = false; },
    hover: function (p) { plane.cv.style.cursor = Math.hypot(p.x - vP.px(handle[0]), p.y - vP.py(handle[1])) < 14 * plane.sc ? 'grab' : 'crosshair'; },
    key: function (e) {
      if (e.key === 'ArrowLeft') { angle = wrapAngle(angle + 0.03); draw(); return true; }
      if (e.key === 'ArrowRight') { angle = wrapAngle(angle - 0.03); draw(); return true; }
    }
  });
  function setFrom(p) {
    var a = Math.atan2(vP.iy(p.y) - mu[1], vP.ix(p.x) - mu[0]);
    a = wrapAngle(a);
    var d = Math.abs(wrapAngle(a - pc1 + Math.PI / 2) - Math.PI / 2);
    angle = d < 0.06 ? wrapAngle(pc1) : a;
    draw();
  }
  M.pointer(curveSt.cv, {
    down: function (p) { tw.stop(); angle = wrapAngle(M.clamp(vC.ix(p.x), 0, 180) * Math.PI / 180); draw(); },
    drag: function (p) { angle = wrapAngle(M.clamp(vC.ix(p.x), 0, 179.9) * Math.PI / 180); draw(); }
  });

  loadPCA();
  loadRandom();
  modeT.set(mode, true);
  plane.draw = draw; curveSt.draw = draw; flat.draw = draw; dist.draw = draw;
  sync();
  plane.request();
  return { destroy: function () { tw.stop(); } };
});
