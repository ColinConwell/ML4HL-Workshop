/* ══════════════════════════════════════════════════════════════════
   Figure 4 — Non-linear regression by piecewise segments and splines
   Left: a smooth curve (or developmental data) tiled by K piecewise
   linear segments or cubic B-splines joined at knots.
   Right: mean squared error vs segment count K on log-log axes, showing
   resolution-limited power-law scaling (MSE ~ K^-4) and the noise floor.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('spline_scaling', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Bending the line: piecewise segments and splines',
    hint: 'Press play or scrub K &middot; switch linear vs cubic spline &middot; toggle adaptive knots'
  });

  var real = cfg.data && cfg.data.x;
  var seed = cfg.seed || 7;
  var LADDER = [1, 2, 3, 4, 6, 8, 10, 14, 20, 28, 40, 56];
  var degree = cfg.degree || 'linear', placement = cfg.placement || 'uniform';
  var xs, ys, xDom, yDom, denseX, targetY, curvCDF, fits, trainErrs, refErrs, slopePow;

  function smoothTarget(u) {
    /* Multi-scale smooth curve from ScalingPessimism theory_media.py (on u in [0, 1]). */
    return Math.sin(2 * Math.PI * u) + 0.45 * Math.sin(6 * Math.PI * u + 0.6) + 0.2 * Math.cos(11 * Math.PI * u);
  }

  function buildCurvatureCDF(gridU, fn) {
    var m = gridU.length, c = new Float64Array(m);
    for (var i = 1; i < m - 1; i++) {
      var d2 = Math.abs(fn(gridU[i + 1]) - 2 * fn(gridU[i]) + fn(gridU[i - 1]));
      c[i] = Math.pow(d2, 1 / 3) + 0.08;
    }
    c[0] = c[1]; c[m - 1] = c[m - 2];
    var cdf = new Float64Array(m), acc = 0;
    for (i = 0; i < m; i++) { acc += c[i]; cdf[i] = acc; }
    for (i = 0; i < m; i++) cdf[i] /= acc || 1;
    return cdf;
  }

  function makeKnots(K) {
    var x0 = xDom[0], x1 = xDom[1], knots = new Float64Array(K + 1);
    knots[0] = x0; knots[K] = x1;
    if (placement === 'uniform' || !curvCDF) {
      for (var k = 1; k < K; k++) knots[k] = x0 + (x1 - x0) * (k / K);
    } else {
      var m = curvCDF.length;
      for (var j = 1; j < K; j++) {
        var q = j / K, lo = 0, hi = m - 1;
        while (lo < hi) { var mid = (lo + hi) >> 1; if (curvCDF[mid] < q) lo = mid + 1; else hi = mid; }
        knots[j] = x0 + (x1 - x0) * (lo / (m - 1));
      }
    }
    return knots;
  }

  function splineBasisRow(x, knots, deg) {
    var K = knots.length - 1, span = (xDom[1] - xDom[0]) || 1;
    var u = (x - xDom[0]) / span;
    if (deg === 'linear') {
      var row = [1, u];
      for (var k = 1; k < K; k++) {
        var uk = (knots[k] - xDom[0]) / span;
        row.push(Math.max(0, u - uk));
      }
      return row;
    } else {
      /* Truncated cubic power basis + ridge stabilization for smooth cubic splines. */
      var row3 = [1, u, u * u, u * u * u];
      for (var j = 1; j < K; j++) {
        var uj = (knots[j] - xDom[0]) / span, d = Math.max(0, u - uj);
        row3.push(d * d * d);
      }
      return row3;
    }
  }

  function load() {
    var rng = M.rng(seed);
    var M_DENSE = 320;
    denseX = new Float64Array(M_DENSE);
    targetY = new Float64Array(M_DENSE);

    if (real) {
      xs = cfg.data.x.slice();
      ys = cfg.data.y.slice();
      var lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs);
      xDom = [lo, hi];
      var ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys), ypad = (yhi - ylo) * 0.12 || 1;
      yDom = [ylo - ypad, yhi + ypad];
      for (var i = 0; i < M_DENSE; i++) denseX[i] = lo + (hi - lo) * (i / (M_DENSE - 1));
      if (cfg.data.curve_x && cfg.data.curve_y) {
        var cx = cfg.data.curve_x, cy = cfg.data.curve_y;
        var interp = function (x) {
          if (x <= cx[0]) return cy[0];
          if (x >= cx[cx.length - 1]) return cy[cy.length - 1];
          for (var j = 0; j < cx.length - 1; j++) {
            if (x >= cx[j] && x <= cx[j + 1]) {
              var t = (x - cx[j]) / ((cx[j + 1] - cx[j]) || 1);
              return cy[j] + t * (cy[j + 1] - cy[j]);
            }
          }
          return cy[cy.length - 1];
        };
        for (i = 0; i < M_DENSE; i++) targetY[i] = interp(denseX[i]);
        curvCDF = buildCurvatureCDF(denseX, interp);
      } else {
        curvCDF = null;
      }
    } else {
      xDom = [0, 1];
      yDom = [-1.95, 1.95];
      var noise = noiseS ? noiseS.get() : (cfg.noise == null ? 0 : cfg.noise);
      for (var g = 0; g < M_DENSE; g++) {
        denseX[g] = g / (M_DENSE - 1);
        targetY[g] = smoothTarget(denseX[g]);
      }
      curvCDF = buildCurvatureCDF(denseX, smoothTarget);
      var nPts = 96;
      xs = []; ys = [];
      for (var p = 0; p < nPts; p++) {
        var x = (p + 0.5) / nPts;
        xs.push(x);
        ys.push(smoothTarget(x) + (noise > 0 ? rng.normal(0, noise) : 0));
      }
    }
    solveAll();
  }

  function solveAll() {
    fits = []; trainErrs = []; refErrs = [];
    var noise = (!real && noiseS) ? noiseS.get() : (cfg.noise || 0);
    var pureInterp = !real && noise === 0 && degree === 'linear';

    for (var s = 0; s < LADDER.length; s++) {
      var K = LADDER[s], knots = makeKnots(K);
      var evalFn;
      if (pureInterp) {
        /* Direct knot interpolation as in ScalingPessimism resolution scaling. */
        var ky = new Float64Array(K + 1);
        for (var k = 0; k <= K; k++) ky[k] = smoothTarget(knots[k]);
        evalFn = (function (kn, kv) {
          return function (x) {
            if (x <= kn[0]) return kv[0];
            if (x >= kn[kn.length - 1]) return kv[kv.length - 1];
            for (var j = 0; j < kn.length - 1; j++) {
              if (x >= kn[j] && x <= kn[j + 1]) {
                var t = (x - kn[j]) / ((kn[j + 1] - kn[j]) || 1e-9);
                return kv[j] + t * (kv[j + 1] - kv[j]);
              }
            }
            return kv[kv.length - 1];
          };
        })(knots, ky);
      } else {
        var my = M.mean(ys), sy = M.std(ys) || 1;
        var Phi = xs.map(function (x) { return splineBasisRow(x, knots, degree); });
        var lam = degree === 'cubic' ? 1e-5 : 1e-7;
        var w = M.lstsq(Phi, ys.map(function (y) { return (y - my) / sy; }), lam);
        evalFn = (function (w, kn, deg, my, sy) {
          return function (x) {
            var b = splineBasisRow(x, kn, deg), sum = 0;
            for (var j = 0; j < b.length; j++) sum += w[j] * b[j];
            return my + sy * sum;
          };
        })(w, knots, degree, my, sy);
      }
      var knotY = new Float64Array(K + 1);
      for (var kj = 0; kj <= K; kj++) knotY[kj] = evalFn(knots[kj]);
      var curvePred = new Float64Array(denseX.length);
      var mseRef = 0;
      for (var i = 0; i < denseX.length; i++) {
        curvePred[i] = evalFn(denseX[i]);
        var d = curvePred[i] - targetY[i];
        mseRef += d * d;
      }
      mseRef /= denseX.length;
      var mseTr = 0;
      for (i = 0; i < xs.length; i++) {
        var r = evalFn(xs[i]) - ys[i];
        mseTr += r * r;
      }
      mseTr /= xs.length;
      fits.push({ K: K, knots: knots, knotY: knotY, curve: curvePred });
      trainErrs.push(Math.max(1e-7, mseTr));
      refErrs.push(Math.max(1e-7, real && !cfg.data.curve_x ? mseTr : mseRef));
    }

    /* Power-law slope over K >= 6 on the reference curve. */
    var tailIdx = [];
    for (var i2 = 0; i2 < LADDER.length; i2++) if (LADDER[i2] >= 6) tailIdx.push(i2);
    var lx = tailIdx.map(function (i) { return Math.log(LADDER[i]); });
    var ly = tailIdx.map(function (i) { return Math.log(refErrs[i]); });
    slopePow = M.linfit(lx, ly);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.35);
  M.mini(split.a, 'piecewise spline fit & knot locations');
  M.mini(split.b, 'error vs segments K (log-log scaling)');
  var hf = function (w) { return M.clamp(w * 0.72, 230, 340); };
  var left = M.stage(split.a, { height: hf, ref: 460, label: 'Target curve or data fitted by a K-segment spline with knots' });
  var right = M.stage(split.b, { height: hf, ref: 360, label: 'Log-log plot of mean squared error versus segment count K' });

  var legend = M.legend(root, []);
  var tr = M.transport(root, {
    n: LADDER.length, fps: 4,
    fmt: function (i) { return LADDER[i] + (LADDER[i] === 1 ? ' segment (line)' : ' segments'); },
    onFrame: function () { drawAll(); }
  });
  var row1 = M.row(root);
  var degT = M.toggle(row1, {
    label: 'Basis', value: degree,
    options: [['linear', 'piecewise linear'], ['cubic', 'cubic spline']],
    onChange: function (v) { degree = v; solveAll(); drawAll(); }
  });
  var placeT = M.toggle(row1, {
    label: 'Knots', value: placement,
    options: [['uniform', 'uniform'], ['adaptive', 'curvature-weighted']],
    onChange: function (v) { placement = v; solveAll(); drawAll(); }
  });
  var noiseS = null;
  if (!real) {
    noiseS = M.slider(row1, {
      label: 'Noise', min: 0, max: 0.45, step: 0.03, value: cfg.noise == null ? 0 : cfg.noise, accent: 'orange',
      fmt: function (v) { return v.toFixed(2); },
      onInput: function () { load(); drawAll(); }
    });
  }
  M.spacer(row1);
  M.resetButton(row1, function () {
    tr.pause();
    degree = cfg.degree || 'linear'; degT.set(degree, true);
    placement = cfg.placement || 'uniform'; placeT.set(placement, true);
    if (noiseS) noiseS.set(cfg.noise == null ? 0 : cfg.noise, true);
    load();
    tr.show(cfg.step_idx == null ? 3 : cfg.step_idx);
    drawAll();
  });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var vL = M.view(left, { x: [0, 1], y: [-1, 1], pad: real ? { l: 52, r: 12, t: 12, b: 38 } : { l: 38, r: 12, t: 12, b: 34 } });
  var vR = M.view(right, { x: [0, 1], y: [0, 1], pad: { l: 50, r: 14, t: 12, b: 38 } });

  function drawLeft(f) {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain(xDom, yDom);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, {
      xlabel: real ? (cfg.data.xlabel || 'x') : 'input x',
      ylabel: real ? (cfg.data.ylabel || 'y') : 'target y',
      ny: 4
    });

    ctx.save(); vL.clip(ctx);
    var noise = (!real && noiseS) ? noiseS.get() : 0;

    /* Shaded approximation gap between spline and underlying curve. */
    if (!real || cfg.data.curve_x) {
      ctx.fillStyle = M.rgba(C.blue, 0.10);
      ctx.beginPath();
      for (var i = 0; i < denseX.length; i++) ctx[i ? 'lineTo' : 'moveTo'](vL.px(denseX[i]), vL.py(targetY[i]));
      for (i = denseX.length - 1; i >= 0; i--) ctx.lineTo(vL.px(denseX[i]), vL.py(f.curve[i]));
      ctx.closePath(); ctx.fill();

      ctx.strokeStyle = real ? M.rgba(C.green, 0.85) : C.ink;
      ctx.lineWidth = 1.6 * s.sc;
      if (real) ctx.setLineDash([5 * s.sc, 3 * s.sc]);
      ctx.beginPath();
      for (i = 0; i < denseX.length; i++) ctx[i ? 'lineTo' : 'moveTo'](vL.px(denseX[i]), vL.py(targetY[i]));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    /* Data points when in real mode or when noise > 0. */
    if (real || noise > 0) {
      for (var p = 0; p < xs.length; p++) {
        M.dot(ctx, vL.px(xs[p]), vL.py(ys[p]), 3 * s.sc, M.rgba(C.ink, real ? 0.45 : 0.35));
      }
    }

    /* Vertical knot ticks along bottom axis. */
    ctx.strokeStyle = M.rgba(C.orange, 0.45);
    ctx.lineWidth = 1;
    for (var k = 0; k < f.knots.length; k++) {
      var kx = vL.px(f.knots[k]);
      ctx.beginPath(); ctx.moveTo(kx, vL.B); ctx.lineTo(kx, vL.B - 7 * s.sc); ctx.stroke();
    }

    /* Fitted spline curve. */
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.3 * s.sc;
    ctx.beginPath();
    for (var j = 0; j < denseX.length; j++) {
      ctx[j ? 'lineTo' : 'moveTo'](vL.px(denseX[j]), vL.py(M.clamp(f.curve[j], yDom[0] - 2, yDom[1] + 2)));
    }
    ctx.stroke();

    /* Knot circles on the spline when K <= 28. */
    if (f.K <= 28) {
      for (k = 0; k < f.knots.length; k++) {
        M.dot(ctx, vL.px(f.knots[k]), vL.py(f.knotY[k]), 3.5 * s.sc, '#fff', C.blue, 1.6 * s.sc);
      }
    }
    ctx.restore();

    M.pill(ctx, s, f.K + (f.K === 1 ? ' segment (2 knots)' : ' segments (' + (f.K + 1) + ' knots)'),
      vL.L + 8 * s.sc, vL.T + 12 * s.sc, { color: C.blue, size: 9.5 });
  }

  function drawRight(idx) {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    var lx0 = Math.log10(LADDER[0]) - 0.08, lx1 = Math.log10(LADDER[LADDER.length - 1]) + 0.08;
    var allE = refErrs.concat(trainErrs);
    var ly0 = Math.floor(Math.log10(Math.min.apply(null, allE)) - 0.3);
    var ly1 = Math.ceil(Math.log10(Math.max.apply(null, allE)) + 0.25);
    vR.setDomain([lx0, lx1], [ly0, ly1]);
    M.clear(ctx, s);
    var xticks = [1, 2, 4, 8, 16, 32, 56].map(function (k) { return Math.log10(k); });
    M.axes(ctx, vR, s, {
      xlabel: 'segments K (log scale)',
      ylabel: 'MSE (log10)',
      xticks: xticks,
      xfmt: function (v) { return String(Math.round(Math.pow(10, v))); },
      yfmt: function (v) { return '10^' + M.f(v, 0); },
      ny: 4
    });

    ctx.save(); vR.clip(ctx);

    /* Power-law fit line on log-log axes. */
    ctx.strokeStyle = M.rgba(C.red, 0.75); ctx.lineWidth = 1.6 * s.sc; ctx.setLineDash([5 * s.sc, 4 * s.sc]);
    ctx.beginPath();
    [LADDER[0], LADDER[LADDER.length - 1]].forEach(function (K, j) {
      var predLogE = (slopePow.m * Math.log(K) + slopePow.b) / Math.LN10;
      ctx[j ? 'lineTo' : 'moveTo'](vR.px(Math.log10(K)), vR.py(predLogE));
    });
    ctx.stroke(); ctx.setLineDash([]);

    /* Faint full curve + active prefix up to idx. */
    ctx.strokeStyle = M.rgba(C.blue, 0.25); ctx.lineWidth = 1.4 * s.sc;
    ctx.beginPath();
    LADDER.forEach(function (K, i) { ctx[i ? 'lineTo' : 'moveTo'](vR.px(Math.log10(K)), vR.py(Math.log10(refErrs[i]))); });
    ctx.stroke();

    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.2 * s.sc;
    ctx.beginPath();
    for (var i = 0; i <= idx; i++) {
      ctx[i ? 'lineTo' : 'moveTo'](vR.px(Math.log10(LADDER[i])), vR.py(Math.log10(refErrs[i])));
    }
    ctx.stroke();
    for (i = 0; i <= idx; i++) {
      M.dot(ctx, vR.px(Math.log10(LADDER[i])), vR.py(Math.log10(refErrs[i])), (i === idx ? 5 : 3.2) * s.sc, C.blue, '#fff', 1.4 * s.sc);
    }
    ctx.restore();

    M.pill(ctx, s, 'scaling slope \u2248 K^' + M.f(slopePow.m, 1), vR.L + 8 * s.sc, vR.B - 14 * s.sc, { color: C.red, size: 9.5 });
  }

  function drawAll() {
    if (!fits) return;
    var idx = tr.i, f = fits[idx];
    drawLeft(f);
    drawRight(idx);
    M.setLegend(legend, [
      { color: real ? C.green : C.ink, label: real ? 'age-month mean trajectory' : 'smooth target curve', line: true },
      { color: C.blue, label: (degree === 'linear' ? 'piecewise-linear' : 'cubic') + ' spline (' + f.K + ' segments)', line: true },
      { color: C.red, label: 'power-law fit (MSE \u221d K^' + M.f(slopePow.m, 1) + ')', line: true }
    ]);
    var e = refErrs[idx];
    var msg = f.K === 1
      ? '<span class="o">1 segment is ordinary linear regression: a single straight line misses every bend</span>'
      : f.K <= 4
        ? '<span class="b">a few knots capture the main inflection points without global oscillation</span>'
        : '<span class="g">finer segments tile the curve locally; error falls as a power of K until hitting the noise floor</span>';
    read.innerHTML = '<span class="k">segments</span> ' + f.K +
      '<span class="sep">&middot;</span><span class="k">knots</span> ' + (f.K + 1) +
      '<span class="sep">&middot;</span><span class="k">MSE</span> ' + M.f(e, e > 10 ? 1 : 4) +
      '<span class="sep">&middot;</span><span class="k">power-law slope</span> ' + M.f(slopePow.m, 2) +
      '<span class="sep">&middot;</span>' + msg;
  }

  M.pointer(right.cv, {
    down: function (p) {
      if (!vR.inside(p.x, p.y)) return false;
      tr.pause();
      var lk = vR.ix(p.x), best = 0;
      LADDER.forEach(function (K, i) { if (Math.abs(Math.log10(K) - lk) < Math.abs(Math.log10(LADDER[best]) - lk)) best = i; });
      tr.show(best);
    },
    drag: function (p) {
      var lk = vR.ix(p.x), best = 0;
      LADDER.forEach(function (K, i) { if (Math.abs(Math.log10(K) - lk) < Math.abs(Math.log10(LADDER[best]) - lk)) best = i; });
      tr.show(best);
    },
    hover: function (p) { right.cv.style.cursor = vR.inside(p.x, p.y) ? 'pointer' : ''; }
  });

  load();
  left.draw = drawAll; right.draw = drawAll;
  tr.show(cfg.step_idx == null ? 3 : cfg.step_idx);
  left.request();
  return { destroy: function () { tr.pause(); } };
});
