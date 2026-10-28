/* ══════════════════════════════════════════════════════════════════
   Figure 4 — What categories look like to a machine
   Every example is a dot in feature space; a category is a cloud of dots.
   Shaded "territories" show where each category lives; a dashed overlay
   shows the best a single straight-line classifier can do.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('category_manifolds', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'What categories look like to a machine',
    hint: 'Switch shapes &middot; click to add a point (shift-click for the next category)'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 5, X, y, K, names, dom, lin, terrGrid = null, linGrid = null;
  var showTerr = cfg.territories !== false, showLin = true;

  function simulate() {
    var rng = M.rng(seed), shape = shapeT ? shapeT.get() : (cfg.shape || 'blobs');
    var n = nS ? nS.get() : (cfg.n || 80), noise = noiseS ? noiseS.get() : (cfg.noise == null ? 0.3 : cfg.noise);
    var k = kS ? kS.get() : (cfg.k || 3), d;
    if (shape === 'blobs') d = M.sim.blobs(rng, { k: k, n: n, spread: 0.1 + noise * 0.55 });
    else if (shape === 'moons') d = M.sim.moons(rng, { n: n, noise: 0.03 + noise * 0.35 });
    else if (shape === 'circles') d = M.sim.circles(rng, { n: n, noise: 0.02 + noise * 0.28 });
    else if (shape === 'xor') d = M.sim.xor(rng, { n: n, noise: noise * 0.35 });
    else d = M.sim.spiral(rng, { n: n, k: k, noise: 0.01 + noise * 0.18 });
    X = d.X; y = d.y;
    K = Math.max.apply(null, y) + 1;
    names = M.range(K).map(function (i) { return 'category ' + String.fromCharCode(65 + i); });
    dom = M.bounds(X, 0.14);
  }
  function loadReal() {
    X = cfg.data.X.map(function (p) { return p.slice(); });
    y = cfg.data.y.slice();
    K = Math.max.apply(null, y) + 1;
    names = cfg.data.names || M.range(K).map(function (i) { return 'class ' + i; });
    dom = M.bounds(X, 0.1);
  }

  /* Multinomial logistic regression by gradient descent on standardized features. */
  function fitLinear() {
    var mu = [M.mean(X.map(function (p) { return p[0]; })), M.mean(X.map(function (p) { return p[1]; }))];
    var sd = [M.std(X.map(function (p) { return p[0]; })) || 1, M.std(X.map(function (p) { return p[1]; })) || 1];
    var W = M.range(K).map(function () { return [0, 0, 0]; }), n = X.length;
    var Z = X.map(function (p) { return [(p[0] - mu[0]) / sd[0], (p[1] - mu[1]) / sd[1]]; });
    for (var it = 0; it < 400; it++) {
      var G = M.range(K).map(function () { return [0, 0, 0]; });
      for (var i = 0; i < n; i++) {
        var s = W.map(function (w) { return w[0] * Z[i][0] + w[1] * Z[i][1] + w[2]; });
        var mx = Math.max.apply(null, s), e = s.map(function (v) { return Math.exp(v - mx); }), tot = M.sum(e);
        for (var c = 0; c < K; c++) {
          var g = e[c] / tot - (y[i] === c ? 1 : 0);
          G[c][0] += g * Z[i][0]; G[c][1] += g * Z[i][1]; G[c][2] += g;
        }
      }
      for (c = 0; c < K; c++) for (var j = 0; j < 3; j++) W[c][j] -= 1.5 * (G[c][j] / n + 0.001 * W[c][j]);
    }
    var predict = function (px, py) {
      var a = (px - mu[0]) / sd[0], b = (py - mu[1]) / sd[1], best = 0, bv = -Infinity;
      for (var c = 0; c < K; c++) { var v = W[c][0] * a + W[c][1] * b + W[c][2]; if (v > bv) { bv = v; best = c; } }
      return best;
    };
    var correct = 0;
    for (var i2 = 0; i2 < n; i2++) if (predict(X[i2][0], X[i2][1]) === y[i2]) correct++;
    lin = { predict: predict, acc: correct / n };
  }

  function refresh() {
    if (real) loadReal(); else simulate();
    fitLinear();
    terrGrid = linGrid = null;
    legend && M.setLegend(legend, names.map(function (nm, i) { return { color: M.catColor(i), label: nm }; }));
    draw();
  }
  function refit() { fitLinear(); terrGrid = linGrid = null; draw(); }

  /* ─── DOM ─── */
  var stage = M.stage(root, { aspect: 0.5, minH: 260, maxH: 400, label: 'Scatter plot of categories with shaded territories' });
  var legend = M.legend(root, []);
  var shapeT = null, nS = null, noiseS = null, kS = null;
  if (!real) {
    var r1 = M.row(root);
    shapeT = M.toggle(r1, {
      label: 'Shape', value: cfg.shape || 'blobs',
      options: [['blobs', 'blobs'], ['moons', 'moons'], ['circles', 'rings'], ['xor', 'checkers'], ['spiral', 'spiral']],
      onChange: refresh
    });
    var r2 = M.row(root);
    nS = M.slider(r2, { label: 'Points', min: 20, max: 200, step: 10, value: cfg.n || 80, onChange: refresh, onInput: function () {} });
    noiseS = M.slider(r2, { label: 'Noise', min: 0, max: 1, step: 0.02, value: cfg.noise == null ? 0.3 : cfg.noise, accent: 'orange', fmt: function (v) { return v.toFixed(2); }, onInput: refresh });
    kS = M.slider(r2, { label: 'Categories', min: 2, max: 4, step: 1, value: cfg.k || 3, onInput: refresh });
  }
  var r3 = M.row(root);
  var terrBtn = M.button(r3, 'territories', function () { showTerr = !showTerr; terrBtn.classList.toggle('on', showTerr); draw(); }, { aria: 'Toggle shaded category territories' });
  var linBtn = M.button(r3, 'best straight line', function () { showLin = !showLin; linBtn.classList.toggle('on', showLin); draw(); }, { aria: 'Toggle the best linear classifier' });
  terrBtn.classList.toggle('on', showTerr); linBtn.classList.toggle('on', showLin);
  M.spacer(r3);
  if (!real) M.button(r3, 'new sample', function () { seed++; refresh(); }, { aria: 'Draw a new random sample' });
  M.resetButton(r3, function () {
    seed = cfg.seed || 5;
    if (!real) { shapeT.set(cfg.shape || 'blobs', true); nS.set(cfg.n || 80, true); noiseS.set(cfg.noise == null ? 0.3 : cfg.noise, true); kS.set(cfg.k || 3, true); }
    refresh();
  });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var v = M.view(stage, { x: [0, 1], y: [0, 1], equal: !real, pad: real ? { l: 52, b: 40 } : { l: 16, b: 14, t: 12, r: 12 } });
  var GN = 72;

  function territories() {
    var span = Math.max(v.x1 - v.x0, v.y1 - v.y0), hh = span * 0.06, inv = 1 / (2 * hh * hh);
    var grid = [];
    for (var j = 0; j < GN; j++) for (var i = 0; i < GN; i++) {
      var gx = v.x0 + (i + 0.5) / GN * (v.x1 - v.x0), gy = v.y1 - (j + 0.5) / GN * (v.y1 - v.y0);
      var dens = new Array(K).fill(0);
      for (var p = 0; p < X.length; p++) {
        var dx = X[p][0] - gx, dy = X[p][1] - gy, d2 = dx * dx + dy * dy;
        if (d2 < 9 * hh * hh) dens[y[p]] += Math.exp(-d2 * inv);
      }
      var tot = M.sum(dens), c = M.argmax(dens);
      grid.push([c, tot > 1e-3 ? dens[c] / tot : 0, Math.min(1, tot / 2.5)]);
    }
    return grid;
  }

  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w || !X) return;
    v.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    if (showTerr) {
      if (!terrGrid) terrGrid = territories();
      M.field(ctx, v, GN, GN, function (x, yy, i, j) {
        var g = terrGrid[j * GN + i], col = M.hex(M.catColor(g[0]));
        var a = g[2] * (0.1 + 0.38 * Math.max(0, (g[1] - 1 / K) / (1 - 1 / K)));
        return [col[0], col[1], col[2], Math.round(a * 255)];
      });
    }
    if (real) M.axes(ctx, v, s, { xlabel: cfg.data.xlabel || 'feature 1', ylabel: cfg.data.ylabel || 'feature 2' });
    if (showLin) {
      if (!linGrid) {
        var LN = 120; linGrid = { n: LN, c: new Int8Array(LN * LN) };
        for (var j = 0; j < LN; j++) for (var i = 0; i < LN; i++) {
          linGrid.c[j * LN + i] = lin.predict(v.x0 + (i + 0.5) / LN * (v.x1 - v.x0), v.y1 - (j + 0.5) / LN * (v.y1 - v.y0));
        }
      }
      var LN2 = linGrid.n, cw = (v.R - v.L) / LN2, ch = (v.B - v.T) / LN2;
      ctx.fillStyle = M.rgba(C.ink, 0.55);
      for (j = 0; j < LN2; j++) for (i = 0; i < LN2; i++) {
        var c0 = linGrid.c[j * LN2 + i];
        if ((i + 1 < LN2 && linGrid.c[j * LN2 + i + 1] !== c0) || (j + 1 < LN2 && linGrid.c[(j + 1) * LN2 + i] !== c0)) {
          if ((i + j) % 2 === 0) ctx.fillRect(v.L + i * cw, v.T + j * ch, 1.7 * s.sc, 1.7 * s.sc);
        }
      }
    }
    ctx.save(); v.clip(ctx);
    for (var p = 0; p < X.length; p++) {
      var wrong = showLin && lin.predict(X[p][0], X[p][1]) !== y[p];
      M.dot(ctx, v.px(X[p][0]), v.py(X[p][1]), 3.4 * s.sc, M.catColor(y[p]), wrong ? C.ink : '#fff', wrong ? 1.4 : 0.9);
    }
    ctx.restore();
    updateReadout();
  }

  function updateReadout() {
    var acc = lin.acc, chance = 1 / K;
    var verdict = acc > 0.97
      ? '<span class="g">straight lines split these categories almost perfectly: they are <em>linearly separable</em></span>'
      : acc > chance + 0.5 * (1 - chance)
        ? '<span class="b">straight lines get most of the way; the leftovers live where the clouds overlap</span>'
        : '<span class="o">no straight line can split these: the categories curl around each other</span>';
    read.innerHTML = '<span class="k">' + X.length + ' points, ' + K + ' categories</span>' +
      '<span class="sep">&middot;</span><span class="k">best straight-line classifier</span> ' + Math.round(acc * 100) + '% correct' +
      ' <span class="k">(chance ' + Math.round(chance * 100) + '%)</span><span class="sep">&middot;</span>' + verdict;
  }

  M.pointer(stage.cv, {
    down: function (p, e) {
      if (!v.inside(p.x, p.y)) return false;
      var cls = e.shiftKey ? 1 % K : 0;
      if (e.shiftKey && K > 2) cls = (lastAdded + 1) % K;
      lastAdded = cls;
      X.push([v.ix(p.x), v.iy(p.y)]); y.push(cls);
      refit();
      return false;
    },
    hover: function (p) { stage.cv.style.cursor = v.inside(p.x, p.y) ? 'copy' : ''; }
  });
  var lastAdded = 0;

  refresh();
  stage.draw = draw;
  stage.onResize = function () { terrGrid = linGrid = null; };
  stage.request();
});
