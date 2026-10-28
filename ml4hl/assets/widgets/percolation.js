/* ══════════════════════════════════════════════════════════════════
   Figure 8 — Clusters as connected ink
   Percolation: grow a disc around every point; touching discs merge into
   one cluster. Affinity propagation: points pass messages and vote for
   "exemplars" that best represent them.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('percolation', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Clusters as connected ink',
    hint: 'Slide the disc size &middot; hover a point &middot; switch to exemplar voting'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 8, X, dom, span, D2, n;
  var mode = cfg.mode || 'percolation', hover = -1;

  function load() {
    if (real) X = cfg.data.X.map(function (p) { return p.slice(); });
    else {
      var rng = M.rng(seed), N = cfg.n || 170, k = 4;
      var d = M.sim.blobs(rng, { k: k, n: Math.round(N * 0.92 / k), spread: 0.2 });
      X = d.X;
      for (var i = X.length; i < N; i++) X.push([rng.uniform(-1.4, 1.4), rng.uniform(-1.2, 1.2)]);
    }
    n = X.length;
    dom = M.bounds(X, 0.1);
    span = Math.max(dom.x[1] - dom.x[0], dom.y[1] - dom.y[0]);
    D2 = [];
    for (var a = 0; a < n; a++) { D2.push(new Float64Array(n)); }
    for (a = 0; a < n; a++) for (var b = a + 1; b < n; b++) { var d2 = M.dist2(X[a], X[b]); D2[a][b] = D2[b][a] = d2; }
    curve = null; ap = null;
  }

  /* ─── Percolation (union-find over touching discs) ─── */
  function components(r) {
    var par = M.range(n), lim = 4 * r * r;
    function find(i) { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; }
    for (var a = 0; a < n; a++) for (var b = a + 1; b < n; b++) if (D2[a][b] <= lim) { var ra = find(a), rb = find(b); if (ra !== rb) par[ra] = rb; }
    var size = {}, lab = new Int32Array(n);
    for (a = 0; a < n; a++) { var r0 = find(a); size[r0] = (size[r0] || 0) + 1; }
    var roots = Object.keys(size).filter(function (k) { return size[k] > 1; }).sort(function (p, q) { return size[q] - size[p]; });
    var id = {};
    roots.forEach(function (k, i) { id[k] = i; });
    for (a = 0; a < n; a++) { var rr = find(a); lab[a] = id[rr] == null ? -1 : id[rr]; }
    return { lab: lab, k: roots.length, biggest: roots.length ? size[roots[0]] / n : 0, loners: n - roots.reduce(function (s, k) { return s + size[k]; }, 0) };
  }
  var curve = null;
  function clusterCurve() {
    if (curve) return curve;
    curve = [];
    for (var t = 0; t <= 60; t++) { var e = 0.02 + (0.45 - 0.02) * t / 60; curve.push([e, components(e * span / 2).k]); }
    return curve;
  }

  /* ─── Affinity propagation ─── */
  var ap = null;
  function runAP(pref) {
    var S = D2.map(function (row) { return Array.prototype.map.call(row, function (v) { return -v; }); });
    var off = [];
    for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) if (a !== b) off.push(S[a][b]);
    off.sort(function (p, q) { return p - q; });
    var p = off[Math.floor(off.length / 2)] * pref;
    for (a = 0; a < n; a++) S[a][a] = p;
    var R = S.map(function (r) { return new Float64Array(r.length); }), A = S.map(function (r) { return new Float64Array(r.length); });
    var lam = 0.9, frames = [], stable = 0, prevKey = '';
    for (var it = 0; it < 220; it++) {
      for (var i = 0; i < n; i++) {
        var m1 = -Infinity, m2 = -Infinity, k1 = -1;
        for (var k = 0; k < n; k++) { var v = A[i][k] + S[i][k]; if (v > m1) { m2 = m1; m1 = v; k1 = k; } else if (v > m2) m2 = v; }
        for (k = 0; k < n; k++) R[i][k] = lam * R[i][k] + (1 - lam) * (S[i][k] - (k === k1 ? m2 : m1));
      }
      for (k = 0; k < n; k++) {
        var sp = 0;
        for (i = 0; i < n; i++) if (i !== k) sp += Math.max(0, R[i][k]);
        for (i = 0; i < n; i++) {
          var nv = i === k ? sp : Math.min(0, R[k][k] + sp - Math.max(0, R[i][k]));
          A[i][k] = lam * A[i][k] + (1 - lam) * nv;
        }
      }
      var ex = [];
      for (k = 0; k < n; k++) if (A[k][k] + R[k][k] > 0) ex.push(k);
      var lab = new Int16Array(n);
      for (i = 0; i < n; i++) {
        if (!ex.length) { lab[i] = -1; continue; }
        var best = ex[0];
        for (var e = 1; e < ex.length; e++) if (S[i][ex[e]] > S[i][best]) best = ex[e];
        lab[i] = ex.indexOf(i) >= 0 ? i : best;
      }
      frames.push({ ex: ex, lab: lab });
      var key = ex.join(',');
      stable = key === prevKey ? stable + 1 : 0; prevKey = key;
      if (stable >= 15 && ex.length) break;
    }
    var first = 0;
    while (first < frames.length - 1 && !frames[first].ex.length) first++;
    first = Math.max(0, first - 2);
    return { frames: frames.slice(first), offset: first, converged: stable >= 15 };
  }

  /* ─── DOM ─── */
  var modeRow = M.row(root);
  var modeT = M.toggle(modeRow, { label: 'Method', value: mode, options: [['percolation', 'touching discs'], ['affinity', 'exemplar voting']], onChange: function (v) { mode = v; sync(); } });
  var stage = M.stage(root, { aspect: 0.54, minH: 260, maxH: 420, label: 'Scatter plot with cluster discs or exemplar assignments' });
  var pRow = M.row(root);
  var epsS = M.slider(pRow, { label: 'Disc size', min: 0.02, max: 0.45, step: 0.005, value: cfg.eps || 0.08, accent: 'blue', fmt: function (v) { return v.toFixed(3); }, onInput: function () { draw(); } });
  var aWrap = M.h('div');
  root.appendChild(aWrap);
  var tr = M.transport(aWrap, { fps: 18, n: 1, fmt: function (i, nn) { var o = ap ? ap.offset : 0; return 'round ' + (i + 1 + o) + ' / ' + (nn + o); }, onFrame: function () { draw(); } });
  var aRow = M.row(aWrap);
  var prefS = M.slider(aRow, { label: 'Pickiness', min: 0.1, max: 3, step: 0.05, value: cfg.preference || 1, accent: 'orange', fmt: function (v) { return v.toFixed(2); }, onChange: function () { computeAP(); tr.play(); } });
  var row = M.row(root);
  M.spacer(row);
  if (!real) M.button(row, 'new data', function () { seed++; load(); computeAP(); sync(); }, { aria: 'Draw a new random dataset' });
  M.resetButton(row, function () { tr.pause(); epsS.set(cfg.eps || 0.08, true); prefS.set(cfg.preference || 1, true); seed = cfg.seed || 8; load(); computeAP(); draw(); });
  var read = M.readout(root);

  function computeAP() { ap = runAP(prefS.get()); tr.setN(ap.frames.length); }
  function sync() {
    pRow.style.display = mode === 'percolation' ? '' : 'none';
    aWrap.style.display = mode === 'affinity' ? '' : 'none';
    if (mode === 'affinity' && !ap) computeAP();
    if (mode !== 'affinity') tr.pause();
    draw();
  }

  /* ─── Drawing ─── */
  var v = M.view(stage, { x: [0, 1], y: [0, 1], equal: !real, pad: { l: 12, r: 12, t: 12, b: 12 } });
  var off = document.createElement('canvas');

  function drawPerc(ctx, s) {
    var r = epsS.get() * span / 2, comp = components(r), rp = Math.abs(v.px(r) - v.px(0));
    off.width = stage.cv.width; off.height = stage.cv.height;
    var o = off.getContext('2d');
    o.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
    for (var i = 0; i < n; i++) {
      o.fillStyle = comp.lab[i] < 0 ? '#bdb8ad' : M.catColor(comp.lab[i]);
      o.beginPath(); o.arc(v.px(X[i][0]), v.py(X[i][1]), rp, 0, Math.PI * 2); o.fill();
    }
    ctx.save(); ctx.globalAlpha = 0.2; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(off, 0, 0); ctx.restore();
    ctx.strokeStyle = M.rgba(C.ink, 0.18); ctx.lineWidth = 1;
    ctx.beginPath();
    var lim = 4 * r * r;
    for (var a = 0; a < n; a++) for (var b = a + 1; b < n; b++) if (D2[a][b] <= lim) { ctx.moveTo(v.px(X[a][0]), v.py(X[a][1])); ctx.lineTo(v.px(X[b][0]), v.py(X[b][1])); }
    ctx.stroke();
    for (i = 0; i < n; i++) M.dot(ctx, v.px(X[i][0]), v.py(X[i][1]), 3 * s.sc, comp.lab[i] < 0 ? C.muted : M.catColor(comp.lab[i]), '#fff', 0.8);
    if (hover >= 0) {
      M.dot(ctx, v.px(X[hover][0]), v.py(X[hover][1]), rp, null, C.ink, 1.4 * s.sc);
      M.dot(ctx, v.px(X[hover][0]), v.py(X[hover][1]), rp * 2, null, M.rgba(C.ink, 0.35), 1 * s.sc);
    }
    drawCurve(ctx, s);
    var phrase = comp.k === 0 ? '<span class="o">dust: every point is its own island</span>'
      : comp.biggest > 0.85 ? '<span class="o">one giant blob: the discs are so big everything touches</span>'
      : '<span class="b">' + comp.k + ' clusters of connected ink</span>';
    read.innerHTML = '<span class="k">disc size</span> ' + epsS.get().toFixed(3) +
      '<span class="sep">&middot;</span>' + phrase +
      '<span class="sep">&middot;</span><span class="k">loners</span> ' + comp.loners +
      '<span class="sep">&middot;</span><span class="k">biggest cluster</span> ' + Math.round(comp.biggest * 100) + '% of points';
  }

  function drawCurve(ctx, s) {
    var cv = clusterCurve(), w = 150 * s.sc, h = 52 * s.sc, x0 = v.R - w - 6 * s.sc, y0 = v.B - h - 6 * s.sc;
    var mx = Math.max.apply(null, cv.map(function (c) { return c[1]; })) || 1;
    ctx.save();
    ctx.fillStyle = M.rgba(C.surface, 0.94); ctx.strokeStyle = C.border;
    ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.fill(); ctx.stroke();
    ctx.font = M.font(s, 8.5, 500); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('CLUSTERS VS DISC SIZE', x0 + 6 * s.sc, y0 + 5 * s.sc);
    var ix = function (e) { return x0 + 6 * s.sc + (e - 0.02) / 0.43 * (w - 12 * s.sc); };
    var iy = function (k) { return y0 + h - 6 * s.sc - k / mx * (h - 22 * s.sc); };
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.4 * s.sc; ctx.beginPath();
    cv.forEach(function (c, i) { ctx[i ? 'lineTo' : 'moveTo'](ix(c[0]), iy(c[1])); });
    ctx.stroke();
    var e = epsS.get(), k = components(e * span / 2).k;
    M.dot(ctx, ix(e), iy(k), 3 * s.sc, C.blue, '#fff', 1);
    ctx.restore();
  }

  function drawAP(ctx, s) {
    if (!ap) computeAP();
    var f = ap.frames[tr.i], exIdx = {};
    f.ex.forEach(function (k, j) { exIdx[k] = j; });
    ctx.lineWidth = 1 * s.sc;
    for (var i = 0; i < n; i++) {
      var e = f.lab[i];
      if (e < 0 || e === i) continue;
      ctx.strokeStyle = M.rgba(M.catColor(exIdx[e]), 0.45);
      ctx.beginPath(); ctx.moveTo(v.px(X[i][0]), v.py(X[i][1])); ctx.lineTo(v.px(X[e][0]), v.py(X[e][1])); ctx.stroke();
    }
    for (i = 0; i < n; i++) {
      var lab = f.lab[i];
      M.dot(ctx, v.px(X[i][0]), v.py(X[i][1]), 3 * s.sc, lab < 0 ? C.muted : M.catColor(exIdx[lab]), '#fff', 0.8);
    }
    f.ex.forEach(function (k, j) { M.star(ctx, v.px(X[k][0]), v.py(X[k][1]), 8.5 * s.sc, M.catColor(j), '#fff'); });
    var done = tr.i === ap.frames.length - 1;
    read.innerHTML = '<span class="k">round</span> ' + (tr.i + 1 + ap.offset) +
      '<span class="sep">&middot;</span><span class="b">' + f.ex.length + ' exemplars</span> <span class="k">(stars)</span>' +
      '<span class="sep">&middot;</span>' + (done && ap.converged ? '<span class="g">the vote has settled</span>' : done ? '<span class="o">still undecided after the last round</span>' : '<span class="k">points are still passing messages&hellip;</span>') +
      '<span class="sep">&middot;</span><span class="k">more pickiness, fewer exemplars</span>';
  }

  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w || !X) return;
    v.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    if (mode === 'percolation') drawPerc(ctx, s); else drawAP(ctx, s);
  }

  M.pointer(stage.cv, {
    hover: function (p) {
      var best = -1, bd = 14 * stage.sc;
      for (var i = 0; i < n; i++) { var d = Math.hypot(v.px(X[i][0]) - p.x, v.py(X[i][1]) - p.y); if (d < bd) { bd = d; best = i; } }
      if (best !== hover) { hover = best; if (mode === 'percolation') draw(); }
    },
    leave: function () { hover = -1; draw(); },
    down: function () { return false; }
  });

  load();
  modeT.set(mode, true);
  stage.draw = draw;
  sync();
  return { destroy: function () { tr.pause(); } };
});
