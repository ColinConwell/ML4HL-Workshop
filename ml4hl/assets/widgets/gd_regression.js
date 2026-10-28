/* ══════════════════════════════════════════════════════════════════
   Figure 3 — Fitting a line by gradient descent
   Left: the data and the current line. Right: the error surface over
   (slope, intercept), with the descent path walking toward the minimum.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('gd_regression', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Fitting a line by gradient descent',
    hint: 'Press play &middot; click the error map to pick a new starting line'
  });

  var real = cfg.data && cfg.data.x;
  var seed = cfg.seed || 11, STEPS = cfg.steps || 220;
  var xs, ys, mx = 0, my = 0, sx = 1, sy = 1, opt, mDom, bDom, start, frames, surf;

  function load() {
    var rx, ry;
    if (real) { rx = cfg.data.x; ry = cfg.data.y; }
    else {
      var d = M.sim.linear(M.rng(seed), { n: cfg.n || 36, noise: cfg.noise == null ? 0.5 : cfg.noise, slope: 0.8, intercept: 0.5 });
      rx = d.x; ry = d.y;
    }
    if (real) { mx = M.mean(rx); my = M.mean(ry); sx = M.std(rx) || 1; sy = M.std(ry) || 1; }
    xs = rx.map(function (x) { return (x - mx) / sx; });
    ys = ry.map(function (y) { return (y - my) / sy; });
    opt = M.linfit(xs, ys);
    mDom = [opt.m - 2.4, opt.m + 2.4];
    bDom = [opt.b - 2.4, opt.b + 2.4];
    start = { m: opt.m - 1.9, b: opt.b + 1.7 };
    surf = null;
  }

  function mse(m, b) { var s = 0; for (var i = 0; i < xs.length; i++) { var r = m * xs[i] + b - ys[i]; s += r * r; } return s / xs.length; }
  var best = function () { return mse(opt.m, opt.b); };

  function descend() {
    var lr = lrSlider.get(), m = start.m, b = start.b, out = [{ m: m, b: b, e: mse(m, b) }];
    for (var t = 0; t < STEPS; t++) {
      var gm = 0, gb = 0;
      for (var i = 0; i < xs.length; i++) { var r = m * xs[i] + b - ys[i]; gm += r * xs[i]; gb += r; }
      m -= lr * 2 * gm / xs.length; b -= lr * 2 * gb / xs.length;
      var e = mse(m, b);
      if (!isFinite(e) || e > 1e6) break;
      out.push({ m: m, b: b, e: e });
    }
    frames = out;
    tr.setN(frames.length, true);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, 'the data and the current line');
  M.mini(split.b, 'error for every possible line');
  var hfn = function (w) { return M.clamp(w * 0.82, 210, 340); };
  var left = M.stage(split.a, { height: hfn, ref: 420, label: 'Data with the line at the current gradient-descent step' });
  var right = M.stage(split.b, { height: hfn, ref: 420, label: 'Error surface over slope and intercept with the descent path' });
  var tr = M.transport(root, {
    fps: 24, n: 1,
    fmt: function (i, n) { return 'step ' + i + ' / ' + (n - 1); },
    stride: function (i) { return i < 30 ? 1 : i < 90 ? 2 : 4; },
    onFrame: function () { drawAll(); }
  });
  var row = M.row(root);
  var lrSlider = M.slider(row, {
    label: 'Step size', min: 0.005, max: 0.6, value: cfg.lr || 0.12, log: true, accent: 'green',
    fmt: function (v) { return v.toFixed(3); },
    onInput: function () { descend(); drawAll(); }
  });
  M.spacer(row);
  if (!real) M.button(row, 'new data', function () { seed++; load(); descend(); tr.show(0); }, { aria: 'Draw a new random dataset' });
  M.resetButton(row, function () { tr.pause(); load(); lrSlider.set(cfg.lr || 0.12, true); descend(); tr.show(0); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var vL = M.view(left, { x: [0, 1], y: [0, 1], pad: real ? { l: 50, b: 38 } : { l: 34, b: 30 } });
  var vR = M.view(right, { x: [0, 1], y: [0, 1], pad: { l: 46, b: 38 } });

  function toRaw(m, b) {
    /* Convert a standardized-space line into original data units. */
    var slope = m * sy / sx;
    return { m: slope, b: my + sy * b - slope * mx };
  }

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    var rx = xs.map(function (x) { return x * sx + mx; }), ry = ys.map(function (y) { return y * sy + my; });
    var bb = M.bounds(rx.map(function (x, i) { return [x, ry[i]]; }), 0.12);
    vL.setDomain(bb.x, bb.y);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, { xlabels: !!real, ylabels: !!real, xlabel: real ? cfg.data.xlabel || 'x' : 'x', ylabel: real ? cfg.data.ylabel || 'y' : 'y' });
    ctx.save(); vL.clip(ctx);
    var i = tr.i;
    for (var k = Math.max(0, i - 8); k < i; k++) {
      var g = toRaw(frames[k].m, frames[k].b);
      ctx.strokeStyle = M.rgba(C.blue, 0.06 + 0.05 * (k - i + 8));
      ctx.lineWidth = 1.2 * s.sc;
      ctx.beginPath(); ctx.moveTo(vL.px(bb.x[0]), vL.py(g.m * bb.x[0] + g.b)); ctx.lineTo(vL.px(bb.x[1]), vL.py(g.m * bb.x[1] + g.b)); ctx.stroke();
    }
    rx.forEach(function (x, j) { M.dot(ctx, vL.px(x), vL.py(ry[j]), 3.3 * s.sc, C.ink, '#fff', 1); });
    var cur = toRaw(frames[i].m, frames[i].b);
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.4 * s.sc;
    ctx.beginPath(); ctx.moveTo(vL.px(bb.x[0]), vL.py(cur.m * bb.x[0] + cur.b)); ctx.lineTo(vL.px(bb.x[1]), vL.py(cur.m * bb.x[1] + cur.b)); ctx.stroke();
    ctx.restore();
  }

  function buildSurface() {
    /* Log-error grid, cached until the data changes. */
    var n = 90, vals = new Float64Array(n * n), lo = Infinity, hi = -Infinity;
    for (var j = 0; j < n; j++) for (var i = 0; i < n; i++) {
      var m = mDom[0] + (i + 0.5) / n * (mDom[1] - mDom[0]);
      var b = bDom[1] - (j + 0.5) / n * (bDom[1] - bDom[0]);
      var e = Math.log(mse(m, b) + 1e-3);
      vals[j * n + i] = e; lo = Math.min(lo, e); hi = Math.max(hi, e);
    }
    surf = { n: n, vals: vals, lo: lo, hi: hi };
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    if (!surf) buildSurface();
    vR.setDomain(mDom, bDom);
    M.clear(ctx, s);
    var n = surf.n;
    M.field(ctx, vR, n, n, function (x, y, i, j) {
      var t = (surf.vals[j * n + i] - surf.lo) / (surf.hi - surf.lo);
      var band = Math.floor(t * 14) % 2 ? 0.94 : 1;
      var c = M.magma(0.1 + 0.85 * t);
      return [c[0] * band, c[1] * band, c[2] * band, 255];
    });
    M.axes(ctx, vR, s, { grid: false, xlabel: real ? 'slope (standardized)' : 'slope', ylabel: real ? 'intercept (standardized)' : 'intercept' });

    var i = tr.i;
    ctx.save(); vR.clip(ctx);
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1.6 * s.sc;
    ctx.beginPath();
    for (var k = 0; k <= i; k++) ctx[k ? 'lineTo' : 'moveTo'](vR.px(frames[k].m), vR.py(frames[k].b));
    ctx.stroke();
    for (k = 0; k <= i; k += Math.max(1, Math.round(i / 40))) M.dot(ctx, vR.px(frames[k].m), vR.py(frames[k].b), 1.8 * s.sc, 'rgba(255,255,255,0.85)');
    M.star(ctx, vR.px(opt.m), vR.py(opt.b), 8 * s.sc, C.gold, '#fff');
    M.dot(ctx, vR.px(start.m), vR.py(start.b), 4 * s.sc, null, '#fff', 1.6 * s.sc);
    M.dot(ctx, vR.px(frames[i].m), vR.py(frames[i].b), 6.5 * s.sc, C.blue, '#fff', 2 * s.sc);
    ctx.restore();
  }

  function drawAll() {
    if (!frames) return;
    drawLeft(); drawRight();
    var f = frames[tr.i], e0 = best(), last = frames[frames.length - 1];
    var raw = toRaw(f.m, f.b);
    var status;
    if (frames.length < STEPS + 1) status = '<span class="r">diverging: the step size is so big each step overshoots further</span>';
    else if (f.e - e0 < 1e-4 * Math.max(1, e0)) status = '<span class="g">arrived at the bottom: this is the least-squares line</span>';
    else if (last.e - e0 > 0.02 * e0 && tr.i === frames.length - 1) status = '<span class="o">ran out of steps before reaching the bottom: try a bigger step size</span>';
    else status = '<span class="b">walking downhill&hellip;</span>';
    read.innerHTML = '<span class="k">slope</span> ' + M.f(raw.m, 2) +
      '<span class="sep">&middot;</span><span class="k">intercept</span> ' + M.f(raw.b, 2) +
      '<span class="sep">&middot;</span><span class="k">error</span> ' + M.f(f.e * (real ? sy * sy : 1), real && sy > 10 ? 0 : 3) +
      ' <span class="k">(best ' + M.f(e0 * (real ? sy * sy : 1), real && sy > 10 ? 0 : 3) + ')</span>' +
      '<span class="sep">&middot;</span>' + status;
  }

  M.pointer(right.cv, {
    down: function (p) {
      if (!vR.inside(p.x, p.y)) return false;
      tr.pause();
      start = { m: vR.ix(p.x), b: vR.iy(p.y) };
      descend(); tr.show(0); drawAll();
    },
    drag: function (p) {
      start = { m: M.clamp(vR.ix(p.x), mDom[0], mDom[1]), b: M.clamp(vR.iy(p.y), bDom[0], bDom[1]) };
      descend(); tr.show(0); drawAll();
    },
    up: function () { tr.play(); },
    hover: function (p) { right.cv.style.cursor = vR.inside(p.x, p.y) ? 'crosshair' : ''; }
  });

  load();
  descend();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: function () { tr.pause(); } };
});
