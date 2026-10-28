/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 4 — Splines as a bank of ReLU hinges
   A piecewise-linear spline is a weighted sum of hinge functions
   v_j * max(0, x - knot_j). Drag the knots by hand, or press play to
   let gradient descent move the knots into the sharpest bends — the
   bridge from fixed-basis spline regression to ReLU neural networks.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('spline_relu_basis', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Splines as a bank of ReLU hinges (from fixed knots to learned features)',
    hint: 'Drag the orange knot pins along the x-axis &middot; press play to optimize knot locations by gradient descent'
  });

  var real = cfg.data && cfg.data.x;
  var seed = cfg.seed || 11;
  var xs, ys, xDom, yDom, knots, weights, mseNow = 0, mseUniform = 0, stepCount = 0;

  function targetCurve(x) {
    /* Developmental S-curve with a sharp threshold + saturation, similar to vocabulary growth. */
    return 0.1 + 1.45 / (1 + Math.exp(-9 * (x - 0.46))) + 0.18 * Math.sin(3.5 * Math.PI * x);
  }

  function load() {
    var rng = M.rng(seed);
    if (real) {
      xs = cfg.data.x.slice();
      ys = cfg.data.y.slice();
      xDom = [Math.min.apply(null, xs), Math.max.apply(null, xs)];
      var ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys), pad = (yhi - ylo) * 0.14 || 1;
      yDom = [ylo - pad, yhi + pad];
    } else {
      xDom = [0, 1];
      yDom = [-0.25, 1.85];
      xs = []; ys = [];
      for (var i = 0; i < 64; i++) {
        var x = (i + 0.5) / 64;
        xs.push(x);
        ys.push(targetCurve(x) + rng.normal(0, 0.05));
      }
    }
    initUniformKnots();
  }

  function initUniformKnots() {
    var K = kS ? kS.get() : (cfg.knots || 4);
    knots = [];
    for (var j = 1; j <= K; j++) {
      knots.push(xDom[0] + (xDom[1] - xDom[0]) * (j / (K + 1)));
    }
    stepCount = 0;
    solveWeights();
    mseUniform = mseNow;
  }

  function solveWeights() {
    knots.sort(function (a, b) { return a - b; });
    var span = (xDom[1] - xDom[0]) || 1, my = M.mean(ys), sy = M.std(ys) || 1;
    var Phi = xs.map(function (x) {
      var u = (x - xDom[0]) / span, row = [1, u];
      for (var j = 0; j < knots.length; j++) {
        row.push(Math.max(0, u - (knots[j] - xDom[0]) / span));
      }
      return row;
    });
    var wStd = M.lstsq(Phi, ys.map(function (y) { return (y - my) / sy; }), 1e-6);
    weights = wStd.map(function (w) { return w * sy; });
    weights[0] += my;
    mseNow = computeMSE(knots, weights);
  }

  function evalAt(x, kn, w) {
    var span = (xDom[1] - xDom[0]) || 1, u = (x - xDom[0]) / span;
    var s = w[0] + w[1] * u;
    for (var j = 0; j < kn.length; j++) {
      s += w[j + 2] * Math.max(0, u - (kn[j] - xDom[0]) / span);
    }
    return s;
  }

  function computeMSE(kn, w) {
    var s = 0;
    for (var i = 0; i < xs.length; i++) {
      var r = evalAt(xs[i], kn, w) - ys[i];
      s += r * r;
    }
    return s / xs.length;
  }

  /* One gradient step on knot positions (coordinate finite-difference + variable projection). */
  function stepKnots() {
    var span = (xDom[1] - xDom[0]) || 1, eps = span * 0.008, lr = span * 0.12;
    var grad = new Float64Array(knots.length);
    var base = mseNow;
    for (var j = 0; j < knots.length; j++) {
      var orig = knots[j];
      knots[j] = M.clamp(orig + eps, xDom[0] + span * 0.04, xDom[1] - span * 0.04);
      solveWeights();
      var mPlus = mseNow;
      knots[j] = M.clamp(orig - eps, xDom[0] + span * 0.04, xDom[1] - span * 0.04);
      solveWeights();
      var mMinus = mseNow;
      knots[j] = orig;
      grad[j] = (mPlus - mMinus) / (2 * eps);
    }
    var gNorm = Math.sqrt(M.sum(Array.from(grad).map(function (g) { return g * g; }))) || 1;
    var moved = 0;
    for (j = 0; j < knots.length; j++) {
      var next = M.clamp(knots[j] - lr * (grad[j] / Math.max(gNorm, 0.5)), xDom[0] + span * 0.05, xDom[1] - span * 0.05);
      moved += Math.abs(next - knots[j]);
      knots[j] = next;
    }
    solveWeights();
    stepCount++;
    return moved > span * 1e-4 && stepCount < 80;
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.25);
  M.mini(split.a, 'sum of hinges = piecewise-linear curve');
  M.mini(split.b, 'individual ReLU hinge units: v_j · max(0, x − knot_j)');
  var hf = function (w) { return M.clamp(w * 0.72, 230, 340); };
  var left = M.stage(split.a, { height: hf, ref: 440, label: 'Data and piecewise-linear spline with draggable knot handles' });
  var right = M.stage(split.b, { height: hf, ref: 380, label: 'Weighted ReLU hinge basis functions for each knot' });

  var row = M.row(root);
  var playBtn = M.button(row, 'optimize knots (GD)', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Optimize knot positions by gradient descent' });
  M.button(row, 'step knots', function () { pause(); stepKnots(); drawAll(); }, { aria: 'Take one gradient step on the knot positions' });
  var kS = M.slider(row, {
    label: 'Hinges (knots)', min: 1, max: 8, step: 1, value: cfg.knots || 4, accent: 'orange',
    onInput: function () { pause(); initUniformKnots(); drawAll(); }
  });
  M.button(row, 'equal spacing', function () { pause(); initUniformKnots(); drawAll(); }, { aria: 'Reset knots to equal spacing' });
  M.spacer(row);
  M.resetButton(row, function () { pause(); kS.set(cfg.knots || 4, true); load(); drawAll(); });
  var read = M.readout(root);

  var acc = 0;
  var loop = M.loop(function (dt) {
    acc += dt;
    if (acc >= 0.06) {
      acc = 0;
      if (!stepKnots()) { pause(); drawAll(); return false; }
      drawAll();
    }
    return true;
  });
  function play() { acc = 0.06; loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'optimize knots (GD)'; playBtn.classList.remove('on'); }

  /* ─── Drawing ─── */
  var vL = M.view(left, { x: [0, 1], y: [0, 1], pad: real ? { l: 52, r: 12, t: 12, b: 38 } : { l: 38, r: 12, t: 12, b: 34 } });
  var vR = M.view(right, { x: [0, 1], y: [-1, 1], pad: { l: 46, r: 12, t: 12, b: 38 } });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain(xDom, yDom);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, {
      xlabel: real ? (cfg.data.xlabel || 'x') : 'input x (drag the orange knot pins)',
      ylabel: real ? (cfg.data.ylabel || 'y') : 'output y',
      ny: 4
    });

    ctx.save(); vL.clip(ctx);
    for (var i = 0; i < xs.length; i++) {
      M.dot(ctx, vL.px(xs[i]), vL.py(ys[i]), 3.2 * s.sc, M.rgba(C.ink, 0.45));
    }

    /* Dashed vertical lines from each knot to the curve. */
    knots.forEach(function (k, j) {
      var kx = vL.px(k), ky = vL.py(evalAt(k, knots, weights));
      ctx.strokeStyle = M.rgba(M.catColor(j), 0.45);
      ctx.lineWidth = 1.2 * s.sc;
      ctx.setLineDash([3 * s.sc, 3 * s.sc]);
      ctx.beginPath(); ctx.moveTo(kx, vL.B); ctx.lineTo(kx, ky); ctx.stroke();
      ctx.setLineDash([]);
    });

    /* Piecewise-linear spline curve (through xDom[0], knots, xDom[1]). */
    var pts = [xDom[0]].concat(knots).concat([xDom[1]]);
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.5 * s.sc;
    ctx.beginPath();
    pts.forEach(function (x, j) { ctx[j ? 'lineTo' : 'moveTo'](vL.px(x), vL.py(evalAt(x, knots, weights))); });
    ctx.stroke();

    /* Draggable knot handles on the curve and on the bottom rail. */
    knots.forEach(function (k, j) {
      var kx = vL.px(k), ky = vL.py(evalAt(k, knots, weights)), col = M.catColor(j);
      M.dot(ctx, kx, ky, 6.2 * s.sc, '#fff', col, 2.2 * s.sc);
      M.dot(ctx, kx, vL.B - 5 * s.sc, 5.2 * s.sc, col, '#fff', 1.4 * s.sc);
    });
    ctx.restore();
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    var span = (xDom[1] - xDom[0]) || 1;
    var maxH = 0.2;
    knots.forEach(function (k, j) {
      var hAtEnd = Math.abs(weights[j + 2] * (xDom[1] - k) / span);
      if (hAtEnd > maxH) maxH = hAtEnd;
    });
    vR.setDomain(xDom, [-maxH * 1.15, maxH * 1.15]);
    M.clear(ctx, s);
    M.axes(ctx, vR, s, {
      xlabel: real ? (cfg.data.xlabel || 'x') : 'input x (each hinge turns on at its knot)',
      ylabel: 'hinge contribution',
      ny: 4
    });

    ctx.save(); vR.clip(ctx);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(vR.L, vR.py(0)); ctx.lineTo(vR.R, vR.py(0)); ctx.stroke();

    knots.forEach(function (k, j) {
      var col = M.catColor(j), wj = weights[j + 2];
      var endVal = wj * (xDom[1] - k) / span;
      ctx.strokeStyle = col; ctx.lineWidth = 2.2 * s.sc;
      ctx.beginPath();
      ctx.moveTo(vR.px(xDom[0]), vR.py(0));
      ctx.lineTo(vR.px(k), vR.py(0));
      ctx.lineTo(vR.px(xDom[1]), vR.py(endVal));
      ctx.stroke();
      M.dot(ctx, vR.px(k), vR.py(0), 4.5 * s.sc, '#fff', col, 2 * s.sc);
    });
    ctx.restore();
  }

  function drawAll() {
    drawLeft(); drawRight();
    var gain = mseUniform > 0 ? (mseUniform - mseNow) / mseUniform : 0;
    var verdict = stepCount > 0 && gain > 0.03
      ? '<span class="g">moving the knots to high-curvature bends cut MSE by ' + Math.round(gain * 100) + '% vs equal spacing</span>'
      : '<span class="b">each hinge max(0, x &minus; &kappa;) adds one slope change at its knot &kappa; &mdash; a 1-layer ReLU network</span>';
    read.innerHTML = '<span class="k">ReLU hinges</span> ' + knots.length +
      '<span class="sep">&middot;</span><span class="k">MSE</span> ' + M.f(mseNow, mseNow > 10 ? 1 : 4) +
      '<span class="sep">&middot;</span><span class="k">uniform-knot MSE</span> ' + M.f(mseUniform, mseUniform > 10 ? 1 : 4) +
      '<span class="sep">&middot;</span>' + verdict;
  }

  var dragIdx = -1;
  function hitKnot(p) {
    var best = -1, bd = 16 * left.sc;
    knots.forEach(function (k, j) {
      var kx = vL.px(k), ky = vL.py(evalAt(k, knots, weights));
      var d = Math.min(Math.hypot(p.x - kx, p.y - ky), Math.hypot(p.x - kx, p.y - (vL.B - 5 * left.sc)));
      if (d < bd) { bd = d; best = j; }
    });
    return best;
  }

  M.pointer(left.cv, {
    down: function (p) {
      var h = hitKnot(p);
      if (h < 0) return false;
      pause();
      dragIdx = h;
      left.cv.style.cursor = 'grabbing';
    },
    drag: function (p) {
      if (dragIdx < 0) return;
      var span = (xDom[1] - xDom[0]) || 1;
      knots[dragIdx] = M.clamp(vL.ix(p.x), xDom[0] + span * 0.03, xDom[1] - span * 0.03);
      solveWeights();
      drawAll();
    },
    up: function () { dragIdx = -1; left.cv.style.cursor = ''; },
    hover: function (p) { left.cv.style.cursor = hitKnot(p) >= 0 ? 'ew-resize' : ''; }
  });

  load();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
