/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 5 — Why not fit a straight line to 0/1 labels?
   Compare OLS on binary labels (linear probability model, which shoots
   below 0% and above 100% and gets tilted by extreme points) against
   the logistic sigmoid link sigma(w x + b).
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('logistic_link', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Why not fit a straight line to 0 / 1 labels? The sigmoid link',
    hint: 'Slide the extreme point &middot; drag steepness &middot; watch the threshold at y = 0.5'
  });

  var real = cfg.data && cfg.data.x;
  var seed = cfg.seed || 6;
  var xs, ys, names, xlabel;

  function load() {
    if (real) {
      xs = cfg.data.x.slice();
      ys = cfg.data.y.slice();
      names = cfg.data.names || ['class 0', 'class 1'];
      xlabel = cfg.data.xlabel || 'feature x';
    } else {
      var rng = M.rng(seed), n = cfg.n || 24;
      xs = []; ys = [];
      for (var i = 0; i < n; i++) {
        var c = i < n / 2 ? 0 : 1;
        var x = (c ? 0.9 : -0.9) + rng.normal(0, 0.55);
        xs.push(x); ys.push(c);
      }
      names = ['category 0', 'category 1'];
      xlabel = 'feature x';
    }
  }

  function fitLogistic1D(allX, allY) {
    var w = 1.5, b = 0;
    for (var it = 0; it < 120; it++) {
      var gw = 0, gb = 0;
      for (var i = 0; i < allX.length; i++) {
        var p = M.sigmoid(w * allX[i] + b), r = p - allY[i];
        gw += r * allX[i]; gb += r;
      }
      w -= 0.4 * (gw / allX.length + 0.005 * w);
      b -= 0.4 * (gb / allX.length);
    }
    return { w: w, b: b };
  }

  /* ─── DOM ─── */
  var stage = M.stage(root, { aspect: 0.46, minH: 240, maxH: 360, label: '1D binary classification comparing OLS line and logistic sigmoid curve' });
  var legend = M.legend(root, [
    { color: C.orange, label: 'OLS straight line (shoots past 0% and 100%)', line: true },
    { color: C.blue, label: 'logistic sigmoid \u03c3(wx + b) (bounded in 0\u2013100%)', line: true }
  ]);
  var row = M.row(root);
  var extS = M.slider(row, {
    label: 'Extreme point x', min: 1.5, max: 6.5, step: 0.1, value: cfg.extreme || 2.2, accent: 'orange',
    fmt: function (v) { return '+' + v.toFixed(1); },
    onInput: function () { draw(); }
  });
  var steepS = M.slider(row, {
    label: 'Sigmoid steepness', min: 0.5, max: 6, step: 0.1, value: cfg.steepness || 2.4, accent: 'blue',
    fmt: function (v) { return v.toFixed(1); },
    onInput: function (v) { manualSteep = v; draw(); }
  });
  var manualSteep = null;
  M.button(row, 'fit sigmoid', function () { manualSteep = null; draw(); });
  M.spacer(row);
  M.resetButton(row, function () { manualSteep = null; extS.set(cfg.extreme || 2.2, true); steepS.set(cfg.steepness || 2.4, true); load(); draw(); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var v = M.view(stage, { x: [-3, 6], y: [-0.28, 1.28], pad: { l: 50, r: 16, t: 14, b: 36 } });

  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w || !xs) return;
    var ex = extS.get();
    var allX = xs.concat([ex, ex + 0.35]), allY = ys.concat([1, 1]);
    var lo = Math.min(-2.8, Math.min.apply(null, allX) - 0.4), hi = Math.max(6.8, Math.max.apply(null, allX) + 0.4);
    v.setDomain([lo, hi], [-0.28, 1.28]);
    M.clear(ctx, s);
    M.axes(ctx, v, s, {
      xlabel: xlabel, ylabel: 'predicted probability',
      yticks: [0, 0.5, 1], yfmt: function (t) { return Math.round(t * 100) + '%'; }
    });

    ctx.save(); v.clip(ctx);
    /* Shaded out-of-bounds zones (< 0% and > 100%). */
    ctx.fillStyle = M.rgba(C.red, 0.07);
    ctx.fillRect(v.L, v.T, v.R - v.L, v.py(1) - v.T);
    ctx.fillRect(v.L, v.py(0), v.R - v.L, v.B - v.py(0));

    /* 50% decision threshold line. */
    ctx.strokeStyle = M.rgba(C.ink, 0.35); ctx.setLineDash([4 * s.sc, 4 * s.sc]);
    ctx.beginPath(); ctx.moveTo(v.L, v.py(0.5)); ctx.lineTo(v.R, v.py(0.5)); ctx.stroke();
    ctx.setLineDash([]);

    var ols = M.linfit(allX, allY);
    var logFit = fitLogistic1D(allX, allY);
    if (manualSteep != null) logFit.w = manualSteep;
    else steepS.set(M.clamp(logFit.w, 0.5, 6), true);

    /* OLS line and its 50% crossing. */
    ctx.strokeStyle = C.orange; ctx.lineWidth = 2.2 * s.sc;
    ctx.beginPath(); ctx.moveTo(v.px(lo), v.py(ols.m * lo + ols.b)); ctx.lineTo(v.px(hi), v.py(ols.m * hi + ols.b)); ctx.stroke();
    var xCutOLS = (0.5 - ols.b) / (ols.m || 1e-6);
    ctx.strokeStyle = M.rgba(C.orange, 0.7); ctx.setLineDash([3 * s.sc, 3 * s.sc]);
    ctx.beginPath(); ctx.moveTo(v.px(xCutOLS), v.py(0)); ctx.lineTo(v.px(xCutOLS), v.py(0.5)); ctx.stroke();
    ctx.setLineDash([]);
    M.dot(ctx, v.px(xCutOLS), v.py(0.5), 4.5 * s.sc, C.orange, '#fff', 1.4 * s.sc);

    /* Logistic sigmoid curve and its 50% crossing. */
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.5 * s.sc;
    ctx.beginPath();
    for (var i = 0; i <= 240; i++) {
      var xx = lo + (hi - lo) * (i / 240);
      ctx[i ? 'lineTo' : 'moveTo'](v.px(xx), v.py(M.sigmoid(logFit.w * xx + logFit.b)));
    }
    ctx.stroke();
    var xCutLog = -logFit.b / (logFit.w || 1e-6);
    M.dot(ctx, v.px(xCutLog), v.py(0.5), 4.8 * s.sc, C.blue, '#fff', 1.5 * s.sc);

    /* Data points. */
    for (i = 0; i < xs.length; i++) {
      M.dot(ctx, v.px(xs[i]), v.py(ys[i]), 3.8 * s.sc, ys[i] ? C.orange : C.blue, '#fff', 1);
    }
    /* Extreme points highlighted. */
    [ex, ex + 0.35].forEach(function (xp) {
      M.dot(ctx, v.px(xp), v.py(1), 6 * s.sc, '#fff', C.red, 2.2 * s.sc);
    });
    ctx.restore();

    var wrongOLS = 0, wrongLog = 0;
    for (var k = 0; k < allX.length; k++) {
      if ((ols.m * allX[k] + ols.b >= 0.5 ? 1 : 0) !== allY[k]) wrongOLS++;
      if ((M.sigmoid(logFit.w * allX[k] + logFit.b) >= 0.5 ? 1 : 0) !== allY[k]) wrongLog++;
    }
    read.innerHTML = '<span class="k">50% boundary</span> <span class="b">sigmoid x = ' + M.f(xCutLog, 2) + '</span> vs <span class="o">OLS x = ' + M.f(xCutOLS, 2) + '</span>' +
      '<span class="sep">&middot;</span><span class="k">mistakes</span> <span class="b">sigmoid ' + wrongLog + '</span> vs <span class="o">OLS ' + wrongOLS + '</span>' +
      '<span class="sep">&middot;</span><span class="g">easy points far from the boundary saturate at 100% on the sigmoid instead of pulling the boundary</span>';
  }

  load();
  stage.draw = draw;
  stage.request();
});
