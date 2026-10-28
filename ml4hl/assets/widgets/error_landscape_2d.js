/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 1 — 2D error basin with saddle points,
   momentum, and stochastic gradient noise
   Compare plain gradient descent (blue), momentum (green), and SGD
   with mini-batch noise (orange) on a 2-parameter error surface.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('error_landscape_2d', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: '2D error surface: saddle points, momentum, and SGD noise',
    hint: 'Click the map to drop the walkers &middot; press play &middot; compare plain GD, momentum, and noisy SGD'
  });

  var seed = cfg.seed || 5, STEPS = cfg.steps || 160;
  var start = { x: -1.45, y: 1.25 }, surf = null, paths = null;
  var GLOBAL = { x: 0.92, y: -0.78 }, LOCAL = { x: -0.95, y: 0.85 }, SADDLE = { x: -0.08, y: 0.06 };

  function loss(x, y) {
    /* 2D landscape with a shallow local minimum, a central saddle plateau, and a deep global basin. */
    var bowl = 0.18 * (x * x + y * y);
    var g1 = -1.35 * Math.exp(-((x - GLOBAL.x) * (x - GLOBAL.x) + (y - GLOBAL.y) * (y - GLOBAL.y)) / 0.42);
    var g2 = -0.78 * Math.exp(-((x - LOCAL.x) * (x - LOCAL.x) + (y - LOCAL.y) * (y - LOCAL.y)) / 0.24);
    var ridge = 0.42 * Math.exp(-((x + 0.25) * (x + 0.25) / 0.12 + (y - 0.15) * (y - 0.15) / 0.55));
    return bowl + g1 + g2 + ridge + 1.45;
  }

  function grad(x, y) {
    var e = 1e-4;
    return [
      (loss(x + e, y) - loss(x - e, y)) / (2 * e),
      (loss(x, y + e) - loss(x, y - e)) / (2 * e)
    ];
  }

  function simulatePaths() {
    var lr = lrS.get(), mom = momS.get(), noise = noiseS.get(), rng = M.rng(seed * 29 + 7);
    var gd = [{ x: start.x, y: start.y, e: loss(start.x, start.y) }];
    var hb = [{ x: start.x, y: start.y, e: loss(start.x, start.y) }];
    var sgd = [{ x: start.x, y: start.y, e: loss(start.x, start.y) }];
    var vx = 0, vy = 0, svx = 0, svy = 0;

    for (var t = 0; t < STEPS; t++) {
      /* 1. Vanilla GD */
      var p0 = gd[gd.length - 1], g0 = grad(p0.x, p0.y);
      var nx0 = M.clamp(p0.x - lr * g0[0], -2, 2), ny0 = M.clamp(p0.y - lr * g0[1], -2, 2);
      gd.push({ x: nx0, y: ny0, e: loss(nx0, ny0) });

      /* 2. Momentum (heavy ball) */
      var p1 = hb[hb.length - 1], g1 = grad(p1.x, p1.y);
      vx = mom * vx - lr * g1[0];
      vy = mom * vy - lr * g1[1];
      var nx1 = M.clamp(p1.x + vx, -2, 2), ny1 = M.clamp(p1.y + vy, -2, 2);
      hb.push({ x: nx1, y: ny1, e: loss(nx1, ny1) });

      /* 3. Noisy SGD + momentum */
      var p2 = sgd[sgd.length - 1], g2 = grad(p2.x, p2.y);
      svx = 0.65 * svx - lr * (g2[0] + rng.normal(0, noise));
      svy = 0.65 * svy - lr * (g2[1] + rng.normal(0, noise));
      var nx2 = M.clamp(p2.x + svx, -2, 2), ny2 = M.clamp(p2.y + svy, -2, 2);
      sgd.push({ x: nx2, y: ny2, e: loss(nx2, ny2) });
    }
    paths = { gd: gd, hb: hb, sgd: sgd };
    if (tr) tr.setN(STEPS + 1, true);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.35);
  M.mini(split.a, '2D error contour map (two weights)');
  M.mini(split.b, 'error over steps');
  var hf = function (w) { return M.clamp(w * 0.76, 230, 340); };
  var map = M.stage(split.a, { height: hf, ref: 440, label: '2D error surface with trajectories for GD, momentum, and noisy SGD' });
  var chart = M.stage(split.b, { height: hf, ref: 360, label: 'Error over steps for the three optimizers' });

  var legend = M.legend(root, [
    { color: C.blue, label: 'plain gradient descent', line: true },
    { color: C.green, label: 'GD + momentum (carries velocity)', line: true },
    { color: C.orange, label: 'stochastic GD (mini-batch noise)', line: true }
  ]);
  var tr = M.transport(root, {
    n: STEPS + 1, fps: 28,
    fmt: function (i, n) { return 'step ' + i + ' / ' + (n - 1); },
    onFrame: function () { drawAll(); }
  });
  var row1 = M.row(root);
  var lrS = M.slider(row1, {
    label: 'Step size', min: 0.01, max: 0.25, step: 0.005, value: cfg.lr || 0.055, accent: 'blue',
    fmt: function (v) { return v.toFixed(3); },
    onInput: function () { simulatePaths(); drawAll(); }
  });
  var momS = M.slider(row1, {
    label: 'Momentum', min: 0, max: 0.95, step: 0.05, value: cfg.momentum == null ? 0.85 : cfg.momentum, accent: 'green',
    fmt: function (v) { return v.toFixed(2); },
    onInput: function () { simulatePaths(); drawAll(); }
  });
  var noiseS = M.slider(row1, {
    label: 'SGD noise', min: 0, max: 1.2, step: 0.05, value: cfg.sgd_noise == null ? 0.45 : cfg.sgd_noise, accent: 'orange',
    fmt: function (v) { return v.toFixed(2); },
    onInput: function () { simulatePaths(); drawAll(); }
  });
  var row2 = M.row(root);
  M.button(row2, 'start near local trap', function () { tr.pause(); start = { x: -1.45, y: 1.25 }; simulatePaths(); tr.show(0); drawAll(); });
  M.button(row2, 'start on saddle ridge', function () { tr.pause(); start = { x: -0.25, y: 0.15 }; simulatePaths(); tr.show(0); drawAll(); });
  M.spacer(row2);
  M.resetButton(row2, function () {
    tr.pause();
    start = { x: -1.45, y: 1.25 };
    lrS.set(cfg.lr || 0.055, true); momS.set(cfg.momentum == null ? 0.85 : cfg.momentum, true); noiseS.set(cfg.sgd_noise == null ? 0.45 : cfg.sgd_noise, true);
    simulatePaths(); tr.show(0); drawAll();
  });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var vM = M.view(map, { x: [-2, 2], y: [-2, 2], equal: true, pad: { l: 42, r: 12, t: 12, b: 34 } });
  var vC = M.view(chart, { x: [0, STEPS], y: [0, 2], pad: { l: 44, r: 12, t: 12, b: 34 } });

  function buildSurface() {
    var n = 84, vals = new Float64Array(n * n), lo = Infinity, hi = -Infinity;
    for (var j = 0; j < n; j++) for (var i = 0; i < n; i++) {
      var x = -2 + (i + 0.5) / n * 4, y = 2 - (j + 0.5) / n * 4;
      var e = loss(x, y);
      vals[j * n + i] = e; lo = Math.min(lo, e); hi = Math.max(hi, e);
    }
    surf = { n: n, vals: vals, lo: lo, hi: hi };
  }

  function drawMap(idx) {
    var ctx = map.ctx, s = map;
    if (!s.w) return;
    if (!surf) buildSurface();
    vM.setDomain([-2, 2], [-2, 2]);
    M.clear(ctx, s);
    var n = surf.n;
    M.field(ctx, vM, n, n, function (x, y, i, j) {
      var t = (surf.vals[j * n + i] - surf.lo) / (surf.hi - surf.lo);
      var band = Math.floor(t * 16) % 2 ? 0.93 : 1;
      var c = M.magma(0.12 + 0.82 * t);
      return [c[0] * band, c[1] * band, c[2] * band, 255];
    });
    M.axes(ctx, vM, s, { grid: false, xlabel: 'weight 1', ylabel: 'weight 2', ny: 4 });

    ctx.save(); vM.clip(ctx);
    M.dot(ctx, vM.px(LOCAL.x), vM.py(LOCAL.y), 4.5 * s.sc, null, 'rgba(255,255,255,0.75)', 1.5 * s.sc);
    M.star(ctx, vM.px(GLOBAL.x), vM.py(GLOBAL.y), 7.5 * s.sc, C.gold, '#fff');

    [['gd', C.blue], ['sgd', C.orange], ['hb', C.green]].forEach(function (pair) {
      var trj = paths[pair[0]], col = pair[1];
      ctx.strokeStyle = col; ctx.lineWidth = 2 * s.sc;
      ctx.beginPath();
      for (var k = 0; k <= idx; k++) ctx[k ? 'lineTo' : 'moveTo'](vM.px(trj[k].x), vM.py(trj[k].y));
      ctx.stroke();
      M.dot(ctx, vM.px(trj[idx].x), vM.py(trj[idx].y), 5.2 * s.sc, col, '#fff', 1.6 * s.sc);
    });
    M.dot(ctx, vM.px(start.x), vM.py(start.y), 3.6 * s.sc, '#fff', C.ink, 1.4 * s.sc);
    ctx.restore();
  }

  function drawChart(idx) {
    var ctx = chart.ctx, s = chart;
    if (!s.w) return;
    var allE = paths.gd.map(function (p) { return p.e; }).concat(paths.hb.map(function (p) { return p.e; }), paths.sgd.map(function (p) { return p.e; }));
    var lo = Math.min.apply(null, allE) - 0.05, hi = Math.max.apply(null, allE) + 0.08;
    vC.setDomain([0, STEPS], [lo, hi]);
    M.clear(ctx, s);
    M.axes(ctx, vC, s, { xlabel: 'step', ylabel: 'error', ny: 4 });

    ctx.save(); vC.clip(ctx);
    [['gd', C.blue], ['sgd', C.orange], ['hb', C.green]].forEach(function (pair) {
      var trj = paths[pair[0]], col = pair[1];
      ctx.strokeStyle = M.rgba(col, 0.22); ctx.lineWidth = 1.3 * s.sc;
      ctx.beginPath();
      trj.forEach(function (p, k) { ctx[k ? 'lineTo' : 'moveTo'](vC.px(k), vC.py(p.e)); });
      ctx.stroke();

      ctx.strokeStyle = col; ctx.lineWidth = 2 * s.sc;
      ctx.beginPath();
      for (var k = 0; k <= idx; k++) ctx[k ? 'lineTo' : 'moveTo'](vC.px(k), vC.py(trj[k].e));
      ctx.stroke();
      M.dot(ctx, vC.px(idx), vC.py(trj[idx].e), 4 * s.sc, col, '#fff', 1.4 * s.sc);
    });
    ctx.restore();
  }

  function drawAll() {
    if (!paths) return;
    var idx = tr.i;
    drawMap(idx); drawChart(idx);
    var eg = paths.gd[idx].e, em = paths.hb[idx].e, es = paths.sgd[idx].e;
    var verdict = em < eg - 0.15
      ? '<span class="g">momentum carries the walker over the barrier into the deep global basin</span>'
      : '<span class="b">plain GD follows the local slope and stops in the first basin it enters</span>';
    read.innerHTML = '<span class="k">step</span> ' + idx +
      '<span class="sep">&middot;</span><span class="b">plain GD</span> ' + M.f(eg, 2) +
      '<span class="sep">&middot;</span><span class="g">momentum</span> ' + M.f(em, 2) +
      '<span class="sep">&middot;</span><span class="o">noisy SGD</span> ' + M.f(es, 2) +
      '<span class="sep">&middot;</span>' + verdict;
  }

  M.pointer(map.cv, {
    down: function (p) {
      if (!vM.inside(p.x, p.y)) return false;
      tr.pause();
      start = { x: M.clamp(vM.ix(p.x), -1.9, 1.9), y: M.clamp(vM.iy(p.y), -1.9, 1.9) };
      simulatePaths(); tr.show(0); drawAll();
    },
    drag: function (p) {
      start = { x: M.clamp(vM.ix(p.x), -1.9, 1.9), y: M.clamp(vM.iy(p.y), -1.9, 1.9) };
      simulatePaths(); tr.show(0); drawAll();
    },
    up: function () { tr.play(); },
    hover: function (p) { map.cv.style.cursor = vM.inside(p.x, p.y) ? 'crosshair' : ''; }
  });

  simulatePaths();
  map.draw = drawAll; chart.draw = drawAll;
  map.request();
  return { destroy: function () { tr.pause(); } };
});
