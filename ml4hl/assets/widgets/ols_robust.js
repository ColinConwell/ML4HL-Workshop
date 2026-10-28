/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 2 — Outliers, leverage, and L2 (OLS) vs L1
   (median / Huber) regression
   Drag the red outlier point to see how squaring the residuals lets
   one extreme observation pull the OLS line, while absolute-error (L1)
   regression resists the pull.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('ols_robust', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Why squaring matters: outliers and OLS (L2) vs robust (L1) regression',
    hint: 'Drag the red outlier point anywhere on the plot &middot; compare the squared-error and absolute-error lines'
  });

  var real = cfg.data && cfg.data.x;
  var seed = cfg.seed || 7;
  var xs, ys, dom, outPt, olsLine, robLine;

  function load() {
    if (real) {
      xs = cfg.data.x.slice();
      ys = cfg.data.y.slice();
    } else {
      var d = M.sim.linear(M.rng(seed), { n: cfg.n || 26, noise: cfg.noise == null ? 0.45 : cfg.noise, slope: 0.75, intercept: 0.3 });
      xs = d.x; ys = d.y;
    }
    var X = xs.map(function (x, i) { return [x, ys[i]]; });
    dom = M.bounds(X, 0.22);
    resetOutlier();
  }

  function resetOutlier() {
    var xSpan = dom.x[1] - dom.x[0], ySpan = dom.y[1] - dom.y[0];
    outPt = [dom.x[0] + 0.84 * xSpan, dom.y[0] + 0.14 * ySpan];
    refit();
  }

  function fitIRLS(allX, allY) {
    /* Iteratively reweighted least squares for L1 / Huber loss. */
    var fit = M.linfit(allX, allY), m = fit.m, b = fit.b;
    var sy = M.std(allY) || 1, delta = 0.25 * sy;
    for (var it = 0; it < 25; it++) {
      var sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
      for (var i = 0; i < allX.length; i++) {
        var r = Math.abs(allY[i] - (m * allX[i] + b));
        var w = 1 / Math.max(delta, r);
        sw += w; swx += w * allX[i]; swy += w * allY[i];
        swxx += w * allX[i] * allX[i]; swxy += w * allX[i] * allY[i];
      }
      var mx = swx / sw, my = swy / sw;
      var denom = swxx - sw * mx * mx;
      m = denom ? (swxy - sw * mx * my) / denom : 0;
      b = my - m * mx;
    }
    return { m: m, b: b };
  }

  function refit() {
    var allX = xs.concat([outPt[0]]), allY = ys.concat([outPt[1]]);
    olsLine = M.linfit(allX, allY);
    robLine = fitIRLS(allX, allY);
  }

  /* ─── DOM ─── */
  var stage = M.stage(root, { aspect: 0.5, minH: 250, maxH: 390, label: 'Scatter plot with a draggable outlier comparing OLS and robust L1 regression' });
  var legend = M.legend(root, [
    { color: C.blue, label: 'OLS line (minimizes squared residuals)', line: true },
    { color: C.green, label: 'robust L1 line (minimizes absolute residuals)', line: true },
    { color: C.red, label: 'draggable outlier (high leverage)' }
  ]);
  var row = M.row(root);
  var showSq = M.toggle(row, {
    label: 'Outlier penalty', value: 'squares',
    options: [['squares', 'squared (L2)'], ['bars', 'absolute (L1)']],
    onChange: function () { draw(); }
  });
  M.button(row, 'extreme outlier', function () {
    outPt = [dom.x[0] + 0.9 * (dom.x[1] - dom.x[0]), dom.y[0] + 0.04 * (dom.y[1] - dom.y[0])];
    refit(); draw();
  });
  M.button(row, 'inlier (on trend)', function () {
    var clean = M.linfit(xs, ys), x = dom.x[0] + 0.82 * (dom.x[1] - dom.x[0]);
    outPt = [x, clean.m * x + clean.b];
    refit(); draw();
  });
  M.spacer(row);
  M.resetButton(row, function () { load(); draw(); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var v = M.view(stage, { x: [0, 1], y: [0, 1], pad: real ? { l: 54, b: 38 } : { l: 40, b: 32 } });

  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w || !xs) return;
    v.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    M.axes(ctx, v, s, {
      xlabel: real ? (cfg.data.xlabel || 'x') : 'x',
      ylabel: real ? (cfg.data.ylabel || 'y') : 'y',
      xlabels: !!real, ylabels: !!real
    });

    ctx.save(); v.clip(ctx);
    var clean = M.linfit(xs, ys);
    /* Faint dashed reference for the clean data without the outlier. */
    ctx.strokeStyle = M.rgba(C.muted, 0.5); ctx.lineWidth = 1.2 * s.sc; ctx.setLineDash([4 * s.sc, 4 * s.sc]);
    ctx.beginPath(); ctx.moveTo(v.px(dom.x[0]), v.py(clean.m * dom.x[0] + clean.b)); ctx.lineTo(v.px(dom.x[1]), v.py(clean.m * dom.x[1] + clean.b)); ctx.stroke();
    ctx.setLineDash([]);

    /* Residual geometry on the outlier point. */
    var ox = v.px(outPt[0]), oy = v.py(outPt[1]);
    var oyOLS = v.py(olsLine.m * outPt[0] + olsLine.b), oyRob = v.py(robLine.m * outPt[0] + robLine.b);
    if (showSq.get() === 'squares') {
      var side = Math.abs(oy - oyOLS);
      ctx.fillStyle = M.rgba(C.red, 0.12); ctx.strokeStyle = M.rgba(C.red, 0.55); ctx.lineWidth = 1.2;
      ctx.fillRect(ox - side, Math.min(oy, oyOLS), side, side);
      ctx.strokeRect(ox - side, Math.min(oy, oyOLS), side, side);
    } else {
      ctx.strokeStyle = M.rgba(C.green, 0.8); ctx.lineWidth = 2.2 * s.sc;
      ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oyRob); ctx.stroke();
    }

    /* Robust L1 line and OLS line. */
    ctx.strokeStyle = C.green; ctx.lineWidth = 2.3 * s.sc;
    ctx.beginPath(); ctx.moveTo(v.px(dom.x[0]), v.py(robLine.m * dom.x[0] + robLine.b)); ctx.lineTo(v.px(dom.x[1]), v.py(robLine.m * dom.x[1] + robLine.b)); ctx.stroke();

    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.3 * s.sc;
    ctx.beginPath(); ctx.moveTo(v.px(dom.x[0]), v.py(olsLine.m * dom.x[0] + olsLine.b)); ctx.lineTo(v.px(dom.x[1]), v.py(olsLine.m * dom.x[1] + olsLine.b)); ctx.stroke();

    for (var i = 0; i < xs.length; i++) M.dot(ctx, v.px(xs[i]), v.py(ys[i]), 3.5 * s.sc, C.ink, '#fff', 1);

    /* Draggable outlier handle. */
    M.dot(ctx, ox, oy, 8.5 * s.sc, '#fff', C.red, 2.4 * s.sc);
    M.dot(ctx, ox, oy, 4 * s.sc, C.red);
    ctx.restore();

    var pullOLS = Math.abs(olsLine.m - clean.m) / (Math.abs(clean.m) || 1);
    var pullRob = Math.abs(robLine.m - clean.m) / (Math.abs(clean.m) || 1);
    read.innerHTML = '<span class="k">clean slope</span> ' + M.f(clean.m, 2) +
      '<span class="sep">&middot;</span><span class="b">OLS slope</span> ' + M.f(olsLine.m, 2) + ' <span class="k">(' + Math.round(pullOLS * 100) + '% shift)</span>' +
      '<span class="sep">&middot;</span><span class="g">robust L1 slope</span> ' + M.f(robLine.m, 2) + ' <span class="k">(' + Math.round(pullRob * 100) + '% shift)</span>' +
      '<span class="sep">&middot;</span><span class="o">squaring a 5&times; miss makes it 25&times; louder; absolute error keeps it at 5&times;</span>';
  }

  var dragging = false;
  M.pointer(stage.cv, {
    down: function (p) {
      if (!v.inside(p.x, p.y)) return false;
      dragging = true;
      outPt = [M.clamp(v.ix(p.x), dom.x[0], dom.x[1]), M.clamp(v.iy(p.y), dom.y[0], dom.y[1])];
      refit(); draw();
    },
    drag: function (p) {
      if (!dragging) return;
      outPt = [M.clamp(v.ix(p.x), dom.x[0], dom.x[1]), M.clamp(v.iy(p.y), dom.y[0], dom.y[1])];
      refit(); draw();
    },
    up: function () { dragging = false; },
    hover: function (p) {
      var near = Math.hypot(p.x - v.px(outPt[0]), p.y - v.py(outPt[1])) < 16 * stage.sc;
      stage.cv.style.cursor = near ? 'grab' : v.inside(p.x, p.y) ? 'pointer' : '';
    }
  });

  load();
  stage.draw = draw;
  stage.request();
});
