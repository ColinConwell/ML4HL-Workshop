/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 9 — Soft responsibilities & elliptical clusters:
   Gaussian Mixture Model (EM) vs K-Means
   K-means carves space with straight perpendicular bisectors and fails
   on tilted, elongated clusters. Expectation-Maximization (EM) learns
   each cluster's covariance ellipse and assigns soft probabilities.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('gmm_em', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Soft responsibilities and oval clusters: Gaussian Mixture EM vs K-means',
    hint: 'Press play or step EM &middot; compare elliptical Gaussian components with spherical K-means'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 14;
  var X, dom, K = cfg.k || 3, means, covs, pis, resp, kmCents, iter = 0;

  function load() {
    var rng = M.rng(seed);
    if (real) {
      X = cfg.data.X.map(function (p) { return p.slice(); });
    } else {
      X = [];
      /* Three elongated / tilted Gaussian clouds where K-means struggles. */
      var specs = [
        { cx: -0.75, cy: 0.45, sx: 0.55, sy: 0.13, rot: 0.52, n: 45 },
        { cx: 0.35, cy: -0.15, sx: 0.58, sy: 0.14, rot: 0.52, n: 45 },
        { cx: 0.65, cy: 0.72, sx: 0.22, sy: 0.22, rot: 0, n: 30 }
      ];
      specs.forEach(function (sp) {
        var c = Math.cos(sp.rot), s = Math.sin(sp.rot);
        for (var i = 0; i < sp.n; i++) {
          var u = rng.normal(0, sp.sx), v = rng.normal(0, sp.sy);
          X.push([sp.cx + u * c - v * s, sp.cy + u * s + v * c]);
        }
      });
    }
    dom = M.bounds(X, 0.16);
    initModels();
  }

  function initModels() {
    K = kS ? kS.get() : (cfg.k || 3);
    var rng = M.rng(seed * 17 + K);
    means = []; covs = []; pis = [];
    for (var k = 0; k < K; k++) {
      var p = X[rng.int(X.length)];
      means.push([p[0] + rng.normal(0, 0.08), p[1] + rng.normal(0, 0.08)]);
      covs.push([[0.35, 0], [0, 0.35]]);
      pis.push(1 / K);
    }
    resp = X.map(function () { return new Array(K).fill(1 / K); });
    iter = 0;
    eStep();
    runKMeans();
  }

  function gaussPdf(p, mu, S) {
    var dx = p[0] - mu[0], dy = p[1] - mu[1];
    var det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    det = Math.max(det, 1e-6);
    var inv00 = S[1][1] / det, inv11 = S[0][0] / det, inv01 = -S[0][1] / det;
    var q = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy);
    return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
  }

  function eStep() {
    for (var i = 0; i < X.length; i++) {
      var tot = 0;
      for (var k = 0; k < K; k++) {
        resp[i][k] = pis[k] * gaussPdf(X[i], means[k], covs[k]) + 1e-12;
        tot += resp[i][k];
      }
      for (k = 0; k < K; k++) resp[i][k] /= tot;
    }
  }

  function mStep() {
    var n = X.length;
    for (var k = 0; k < K; k++) {
      var nk = 1e-6, mx = 0, my = 0;
      for (var i = 0; i < n; i++) {
        var r = resp[i][k];
        nk += r; mx += r * X[i][0]; my += r * X[i][1];
      }
      mx /= nk; my /= nk;
      means[k] = [mx, my];
      pis[k] = nk / n;
      var sxx = 0.015, syy = 0.015, sxy = 0;
      for (i = 0; i < n; i++) {
        var r2 = resp[i][k], dx = X[i][0] - mx, dy = X[i][1] - my;
        sxx += r2 * dx * dx; syy += r2 * dy * dy; sxy += r2 * dx * dy;
      }
      covs[k] = [[sxx / nk, sxy / nk], [sxy / nk, syy / nk]];
    }
    eStep();
    iter++;
  }

  function runKMeans() {
    var rng = M.rng(seed * 17 + K);
    kmCents = [];
    for (var k = 0; k < K; k++) kmCents.push(X[rng.int(X.length)].slice());
    for (var it = 0; it < 30; it++) {
      var sums = kmCents.map(function () { return [0, 0, 0]; });
      for (var i = 0; i < X.length; i++) {
        var best = 0, bd = Infinity;
        for (var c = 0; c < K; c++) { var d = M.dist2(X[i], kmCents[c]); if (d < bd) { bd = d; best = c; } }
        sums[best][0] += X[i][0]; sums[best][1] += X[i][1]; sums[best][2]++;
      }
      for (c = 0; c < K; c++) if (sums[c][2]) kmCents[c] = [sums[c][0] / sums[c][2], sums[c][1] / sums[c][2]];
    }
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, 'Gaussian mixture EM (soft responsibilities + ellipses)');
  M.mini(split.b, 'hard K-means on the same points (spherical Voronoi cells)');
  var hf = function (w) { return M.clamp(w * 0.78, 220, 320); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: 'Gaussian mixture model with covariance ellipses and soft responsibilities' });
  var right = M.stage(split.b, { height: hf, ref: 400, label: 'K-means hard Voronoi partitioning on the same dataset' });

  var row = M.row(root);
  var playBtn = M.button(row, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Run or pause EM steps' });
  M.button(row, 'step EM', function () { pause(); mStep(); drawAll(); }, { aria: 'Run one Expectation-Maximization step' });
  var kS = M.slider(row, {
    label: 'Components (k)', min: 2, max: 5, step: 1, value: K, accent: 'blue',
    onInput: function () { pause(); initModels(); drawAll(); }
  });
  M.spacer(row);
  if (!real) M.button(row, 'new data', function () { pause(); seed++; load(); drawAll(); });
  M.resetButton(row, function () { pause(); seed = cfg.seed || 14; kS.set(cfg.k || 3, true); load(); drawAll(); });
  var read = M.readout(root);

  var acc = 0;
  var loop = M.loop(function (dt) {
    acc += dt;
    if (acc >= 0.22) {
      acc = 0;
      mStep();
      drawAll();
      if (iter >= 35) { pause(); return false; }
    }
    return true;
  });
  function play() { if (iter >= 35) initModels(); acc = 0.22; loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  /* ─── Drawing ─── */
  var vL = M.view(left, { x: [0, 1], y: [0, 1], equal: true, pad: { l: 12, r: 12, t: 12, b: 12 } });
  var vR = M.view(right, { x: [0, 1], y: [0, 1], equal: true, pad: { l: 12, r: 12, t: 12, b: 12 } });

  function drawEllipse(ctx, v, mu, S, col, sc) {
    var e = M.eigh(S);
    var a = Math.atan2(e.vectors[0][1], e.vectors[0][0]);
    [1.15, 2.1].forEach(function (mult, idx) {
      var rx = Math.sqrt(Math.max(1e-4, e.values[0])) * mult;
      var ry = Math.sqrt(Math.max(1e-4, e.values[1])) * mult;
      ctx.beginPath();
      for (var t = 0; t <= 64; t++) {
        var th = (2 * Math.PI * t) / 64;
        var ux = rx * Math.cos(th), uy = ry * Math.sin(th);
        var px = mu[0] + ux * Math.cos(a) - uy * Math.sin(a);
        var py = mu[1] + ux * Math.sin(a) + uy * Math.cos(a);
        ctx[t ? 'lineTo' : 'moveTo'](v.px(px), v.py(py));
      }
      ctx.closePath();
      if (idx === 0) { ctx.fillStyle = M.rgba(col, 0.14); ctx.fill(); }
      ctx.strokeStyle = M.rgba(col, idx === 0 ? 0.85 : 0.45);
      ctx.lineWidth = (idx === 0 ? 1.8 : 1.1) * sc;
      ctx.stroke();
    });
  }

  function blendColor(r) {
    var cr = 0, cg = 0, cb = 0;
    for (var k = 0; k < K; k++) {
      var rgb = M.hex(M.catColor(k));
      cr += r[k] * rgb[0]; cg += r[k] * rgb[1]; cb += r[k] * rgb[2];
    }
    return M.css([cr, cg, cb]);
  }

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    ctx.save(); vL.clip(ctx);
    for (var k = 0; k < K; k++) drawEllipse(ctx, vL, means[k], covs[k], M.catColor(k), s.sc);
    for (var i = 0; i < X.length; i++) {
      M.dot(ctx, vL.px(X[i][0]), vL.py(X[i][1]), 3.6 * s.sc, blendColor(resp[i]), '#fff', 0.9);
    }
    ctx.restore();
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    vR.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    M.field(ctx, vR, 140, 110, function (x, y) {
      var best = 0, bd = Infinity;
      for (var c = 0; c < K; c++) { var d = M.dist2([x, y], kmCents[c]); if (d < bd) { bd = d; best = c; } }
      var rgb = M.hex(M.catColor(best));
      return [rgb[0], rgb[1], rgb[2], 32];
    });
    ctx.save(); vR.clip(ctx);
    for (var i = 0; i < X.length; i++) {
      var best = 0, bd = Infinity;
      for (var c = 0; c < K; c++) { var d = M.dist2(X[i], kmCents[c]); if (d < bd) { bd = d; best = c; } }
      M.dot(ctx, vR.px(X[i][0]), vR.py(X[i][1]), 3.6 * s.sc, M.catColor(best), '#fff', 0.9);
    }
    kmCents.forEach(function (c, idx) {
      M.dot(ctx, vR.px(c[0]), vR.py(c[1]), 6 * s.sc, '#fff', M.catColor(idx), 2.2 * s.sc);
    });
    ctx.restore();
  }

  function drawAll() {
    drawLeft(); drawRight();
    var ambig = 0;
    for (var i = 0; i < resp.length; i++) if (Math.max.apply(null, resp[i]) < 0.8) ambig++;
    read.innerHTML = '<span class="k">EM round</span> ' + iter +
      '<span class="sep">&middot;</span><span class="k">borderline points (&lt;80% certainty)</span> ' + ambig +
      '<span class="sep">&middot;</span><span class="g">Gaussian mixtures adapt to tilted, elongated clouds and report uncertainty where clusters overlap</span>';
  }

  load();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
