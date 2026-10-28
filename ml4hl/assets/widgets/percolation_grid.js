/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 8 — Bond percolation & critical threshold on a
   2D lattice (adapted from ScalingPessimism theory_grid.html)
   Each lattice bond is open with probability p. Below p_c = 1/2 only
   isolated islands form; at p_c = 1/2 a giant connected component first
   spans the grid.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('percolation_grid', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Lattice percolation: connected clusters and the critical threshold',
    hint: 'Press play or slide open-bond probability p &middot; toggle all clusters vs largest component'
  });

  var seed = cfg.seed || 2026, G = 96, N = G * G;
  var across, down, par, sz, viewMode = cfg.view || 'clusters', curve = null;
  var PAL = [
    [37, 99, 235], [217, 119, 6], [22, 163, 74], [184, 58, 42],
    [106, 76, 147], [15, 118, 110], [219, 39, 119], [8, 145, 178],
    [101, 163, 13], [147, 51, 234], [234, 88, 12], [75, 85, 99]
  ];

  function buildLattice() {
    var rng = M.rng(seed);
    across = new Float64Array(N);
    down = new Float64Array(N);
    par = new Int32Array(N);
    sz = new Int32Array(N);
    for (var i = 0; i < N; i++) { across[i] = rng.next(); down[i] = rng.next(); }
    curve = null;
  }

  function find(i) {
    while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; }
    return i;
  }
  function union(a, b) {
    a = find(a); b = find(b);
    if (a !== b) { if (a < b) par[b] = a; else par[a] = b; }
  }

  function runPercolation(p) {
    var i, x, y;
    for (i = 0; i < N; i++) par[i] = i;
    for (y = 0; y < G; y++) for (x = 0; x < G; x++) {
      i = y * G + x;
      if (x > 0 && across[i] < p) union(i, i - 1);
      if (y > 0 && down[i] < p) union(i, i - G);
    }
    sz.fill(0);
    var best = 0, bestSz = 0, nClusters = 0;
    for (i = 0; i < N; i++) {
      var r = find(i);
      if (sz[r] === 0) nClusters++;
      if (++sz[r] > bestSz) { bestSz = sz[r]; best = r; }
    }
    return { best: best, bestShare: bestSz / N, nClusters: nClusters };
  }

  function buildCurve() {
    if (curve) return curve;
    curve = [];
    for (var k = 0; k <= 50; k++) {
      var p = 0.2 + 0.6 * (k / 50);
      var res = runPercolation(p);
      curve.push({ p: p, share: res.bestShare, k: res.nClusters });
    }
    return curve;
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.2);
  M.mini(split.a, 'bond percolation lattice (critical at p = 0.50)');
  M.mini(split.b, 'largest cluster share vs probability p');
  var hf = function (w) { return M.clamp(w * 0.78, 220, 330); };
  var gridSt = M.stage(split.a, { height: hf, ref: 400, label: '2D bond percolation lattice colored by connected cluster' });
  var chartSt = M.stage(split.b, { height: hf, ref: 360, label: 'Phase transition curve showing largest cluster share versus bond probability p' });

  var STEPS = 61, ready = false, tr;
  tr = M.transport(root, {
    n: STEPS, fps: 12,
    fmt: function (i) { var p = 0.2 + 0.6 * (i / (STEPS - 1)); return 'p = ' + p.toFixed(2); },
    onFrame: function (i) { if (ready) drawAll(i); }
  });
  var row = M.row(root);
  var modeT = M.toggle(row, {
    label: 'Coloring', value: viewMode,
    options: [['clusters', 'all clusters'], ['giant', 'largest cluster only']],
    onChange: function (v) { viewMode = v; drawAll(); }
  });
  M.button(row, 'critical point (p = 0.50)', function () { tr.pause(); tr.show(30); drawAll(); });
  M.spacer(row);
  M.button(row, 'new lattice', function () { tr.pause(); seed++; buildLattice(); drawAll(); });
  M.resetButton(row, function () { tr.pause(); seed = cfg.seed || 2026; viewMode = cfg.view || 'clusters'; modeT.set(viewMode, true); buildLattice(); tr.show(30); drawAll(); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var off = document.createElement('canvas');
  off.width = G; off.height = G;
  var octx = off.getContext('2d'), img = octx.createImageData(G, G);

  function drawGrid(p, res) {
    var ctx = gridSt.ctx, s = gridSt;
    if (!s.w) return;
    M.clear(ctx, s);
    var d = img.data;
    for (var i = 0; i < N; i++) {
      var r = find(i), c;
      if (viewMode === 'giant') {
        c = r === res.best ? [28, 28, 28] : (sz[r] >= 8 ? [190, 186, 178] : [250, 248, 244]);
      } else {
        c = sz[r] >= 8 ? PAL[(Math.imul(r, 2654435761) >>> 28) % PAL.length] : [236, 232, 224];
      }
      d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    var pad = 10 * s.sc, side = Math.min(s.w - 2 * pad, s.h - 2 * pad);
    var x0 = (s.w - side) / 2, y0 = (s.h - side) / 2;
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, x0, y0, side, side);
    ctx.strokeStyle = C.border; ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, side - 1, side - 1);
    ctx.restore();
  }

  var vC = M.view(chartSt, { x: [0.2, 0.8], y: [0, 1], pad: { l: 46, r: 12, t: 14, b: 36 } });
  function drawChart(p, res) {
    var ctx = chartSt.ctx, s = chartSt;
    if (!s.w) return;
    var cv = buildCurve();
    vC.setDomain([0.2, 0.8], [0, 1.04]);
    M.clear(ctx, s);
    M.axes(ctx, vC, s, {
      xlabel: 'open-bond probability p',
      ylabel: 'largest cluster share',
      xticks: [0.2, 0.35, 0.5, 0.65, 0.8],
      yfmt: function (v) { return Math.round(v * 100) + '%'; },
      ny: 4
    });

    ctx.save(); vC.clip(ctx);
    ctx.strokeStyle = M.rgba(C.red, 0.65); ctx.lineWidth = 1.4 * s.sc; ctx.setLineDash([4 * s.sc, 3 * s.sc]);
    ctx.beginPath(); ctx.moveTo(vC.px(0.5), vC.T); ctx.lineTo(vC.px(0.5), vC.B); ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.2 * s.sc;
    ctx.beginPath();
    cv.forEach(function (pt, i) { ctx[i ? 'lineTo' : 'moveTo'](vC.px(pt.p), vC.py(pt.share)); });
    ctx.stroke();

    M.dot(ctx, vC.px(p), vC.py(res.bestShare), 5.5 * s.sc, C.blue, '#fff', 1.8 * s.sc);
    ctx.restore();
    M.pill(ctx, s, 'critical p_c = 0.50', vC.px(0.5), vC.T + 11 * s.sc, { align: 'center', color: C.red, size: 9 });
  }

  function drawAll(idx) {
    if (!ready) return;
    var stepIdx = typeof idx === 'number' ? idx : (tr ? tr.i : 30);
    var p = 0.2 + 0.6 * (stepIdx / (STEPS - 1));
    buildCurve();
    var res = runPercolation(p);
    drawGrid(p, res);
    drawChart(p, res);
    var msg = Math.abs(p - 0.5) < 0.025
      ? '<span class="g">at the critical threshold p_c = 0.50, clusters appear at every scale and first span the lattice</span>'
      : p < 0.5
        ? '<span class="o">subcritical: local links form small isolated islands</span>'
        : '<span class="b">supercritical: most sites merge into one spanning giant component</span>';
    read.innerHTML = '<span class="k">bond probability p</span> ' + p.toFixed(2) +
      '<span class="sep">&middot;</span><span class="k">largest cluster</span> ' + Math.round(res.bestShare * 100) + '% of lattice' +
      '<span class="sep">&middot;</span>' + msg;
  }

  buildLattice();
  ready = true;
  gridSt.draw = function () { drawAll(); };
  chartSt.draw = function () { drawAll(); };
  tr.show(30);
  drawAll(30);
  return { destroy: function () { tr.pause(); } };
});
