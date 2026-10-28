/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 12 — Scree spectrum & live PCA reconstruction
   Slide how many principal components k to keep (from 1 up to d).
   Left: variance explained per component (scree plot) + cumulative share.
   Right: original vs k-component reconstructed profile for a sample item.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('pca_reconstruct', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'PCA scree spectrum and profile reconstruction',
    hint: 'Slide the kept components k &middot; click a bar or cycle examples to see what PCA keeps vs drops'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 9;
  var X, featNames, D, pcaRes, shares, sampleIdx = 0;

  function load() {
    if (real) {
      X = cfg.data.X.map(function (r) { return r.slice(); });
      D = X[0].length;
      featNames = cfg.data.feature_names || M.range(D).map(function (j) { return 'f' + (j + 1); });
    } else {
      var rng = M.rng(seed);
      D = 12;
      featNames = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12'];
      /* 3 strong latent factors + small independent noise across 12 features. */
      var patterns = [
        M.range(D).map(function (j) { return Math.sin(2 * Math.PI * j / D); }),
        M.range(D).map(function (j) { return Math.cos(2 * Math.PI * j / D); }),
        M.range(D).map(function (j) { return j < D / 2 ? 0.7 : -0.7; })
      ];
      X = [];
      for (var i = 0; i < 120; i++) {
        var z1 = rng.normal(0, 1.7), z2 = rng.normal(0, 1.05), z3 = rng.normal(0, 0.65);
        X.push(M.range(D).map(function (j) {
          return z1 * patterns[0][j] + z2 * patterns[1][j] + z3 * patterns[2][j] + rng.normal(0, 0.22);
        }));
      }
    }
    pcaRes = M.pca(X, D);
    var tot = M.sum(pcaRes.values) || 1;
    shares = pcaRes.values.map(function (v) { return Math.max(0, v) / tot; });
    if (kS) { kS.input.max = D; kS.set(Math.min(cfg.k || 3, D), true); }
  }

  function reconstruct(idx, k) {
    var xc = pcaRes.Xc[idx], out = pcaRes.mean.slice();
    for (var c = 0; c < k; c++) {
      var u = pcaRes.components[c], score = 0;
      for (var j = 0; j < D; j++) score += xc[j] * u[j];
      for (j = 0; j < D; j++) out[j] += score * u[j];
    }
    return out;
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, 'scree plot: variance captured by each component');
  M.mini(split.b, 'original vs k-component reconstruction');
  var hf = function (w) { return M.clamp(w * 0.72, 220, 320); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: 'Scree plot of variance explained by each principal component' });
  var right = M.stage(split.b, { height: hf, ref: 400, label: 'Comparison of original feature profile and k-component PCA reconstruction' });

  var legend = M.legend(root, [
    { color: C.blue, label: 'kept components (1..k)' },
    { color: C.faint, label: 'dropped components' },
    { color: C.ink, label: 'original profile', line: true },
    { color: C.orange, label: 'reconstructed from k components', line: true }
  ]);
  var row = M.row(root);
  var kS = M.slider(row, {
    label: 'Kept components (k)', min: 1, max: 12, step: 1, value: cfg.k || 3, accent: 'blue',
    onInput: function () { drawAll(); }
  });
  M.button(row, 'next item', function () { sampleIdx = (sampleIdx + 1) % X.length; drawAll(); });
  M.spacer(row);
  M.resetButton(row, function () { sampleIdx = 0; load(); drawAll(); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var vL = M.view(left, { x: [0.4, 12.6], y: [0, 1], pad: { l: 44, r: 12, t: 14, b: 36 } });
  var vR = M.view(right, { x: [0, 11], y: [-2, 2], pad: { l: 42, r: 12, t: 14, b: 44 } });

  function drawLeft(k) {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    vL.setDomain([0.4, D + 0.6], [0, 1.04]);
    M.clear(ctx, s);
    M.axes(ctx, vL, s, {
      xlabel: 'principal component',
      ylabel: 'share of variance',
      xticks: M.range(D).map(function (i) { return i + 1; }),
      yfmt: function (v) { return Math.round(v * 100) + '%'; },
      ny: 4, grid: 'y'
    });

    ctx.save(); vL.clip(ctx);
    var bw = (vL.R - vL.L) / D * 0.66;
    var cum = 0, cumPts = [];
    for (var i = 0; i < D; i++) {
      cum += shares[i];
      cumPts.push([i + 1, cum]);
      var cx = vL.px(i + 1), top = vL.py(shares[i]);
      ctx.fillStyle = (i < k) ? C.blue : C.faint;
      ctx.fillRect(cx - bw / 2, top, bw, vL.B - top);
    }
    ctx.strokeStyle = C.green; ctx.lineWidth = 2 * s.sc;
    ctx.beginPath();
    cumPts.forEach(function (p, i) { ctx[i ? 'lineTo' : 'moveTo'](vL.px(p[0]), vL.py(p[1])); });
    ctx.stroke();
    cumPts.forEach(function (p, i) {
      M.dot(ctx, vL.px(p[0]), vL.py(p[1]), (i + 1 === k ? 4.8 : 2.8) * s.sc, C.green, '#fff', 1.2 * s.sc);
    });
    ctx.restore();
    M.pill(ctx, s, 'cumulative: ' + M.pct(cumPts[k - 1][1], 0), vL.R - 6 * s.sc, vL.T + 12 * s.sc, { align: 'right', color: C.green, size: 9 });
  }

  function drawRight(k) {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    var orig = X[sampleIdx], rec = reconstruct(sampleIdx, k);
    var allV = orig.concat(rec);
    var lo = Math.min.apply(null, allV) - 0.35, hi = Math.max.apply(null, allV) + 0.35;
    vR.setDomain([-0.3, D - 0.7], [lo, hi]);
    M.clear(ctx, s);
    M.axes(ctx, vR, s, {
      xlabel: 'features (1..' + D + ')',
      ylabel: 'value (z)',
      xlabels: false, ny: 4
    });

    ctx.save(); vR.clip(ctx);
    /* Residual bars between original and reconstruction. */
    ctx.strokeStyle = M.rgba(C.orange, 0.45); ctx.lineWidth = 1.6 * s.sc;
    for (var j = 0; j < D; j++) {
      ctx.beginPath(); ctx.moveTo(vR.px(j), vR.py(orig[j])); ctx.lineTo(vR.px(j), vR.py(rec[j])); ctx.stroke();
    }

    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8 * s.sc;
    ctx.beginPath();
    orig.forEach(function (val, j) { ctx[j ? 'lineTo' : 'moveTo'](vR.px(j), vR.py(val)); });
    ctx.stroke();
    orig.forEach(function (val, j) { M.dot(ctx, vR.px(j), vR.py(val), 3.2 * s.sc, C.ink, '#fff', 0.9); });

    ctx.strokeStyle = C.orange; ctx.lineWidth = 2.2 * s.sc;
    ctx.beginPath();
    rec.forEach(function (val, j) { ctx[j ? 'lineTo' : 'moveTo'](vR.px(j), vR.py(val)); });
    ctx.stroke();
    rec.forEach(function (val, j) { M.dot(ctx, vR.px(j), vR.py(val), 3.6 * s.sc, C.orange, '#fff', 1); });
    ctx.restore();

    /* Feature labels along bottom. */
    ctx.save();
    ctx.font = M.font(s, 8); ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (j = 0; j < D; j++) {
      ctx.save();
      ctx.translate(vR.px(j), vR.B + 5 * s.sc);
      ctx.rotate(-Math.PI / 3);
      ctx.fillText(String(featNames[j]).slice(0, 10), 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  function drawAll() {
    if (!pcaRes) return;
    var k = M.clamp(kS.get(), 1, D);
    drawLeft(k); drawRight(k);
    var cumShare = M.sum(shares.slice(0, k));
    var orig = X[sampleIdx], rec = reconstruct(sampleIdx, k);
    var mse = M.mean(orig.map(function (v, j) { var d = v - rec[j]; return d * d; }));
    read.innerHTML = '<span class="k">kept</span> ' + k + ' / ' + D + ' components' +
      '<span class="sep">&middot;</span><span class="g">variance retained</span> ' + M.pct(cumShare, 1) +
      '<span class="sep">&middot;</span><span class="o">reconstruction MSE</span> ' + M.f(mse, 3) +
      '<span class="sep">&middot;</span><span class="b">the ' + (cumShare > 0.8 ? 'first ' + k + ' components already capture the main profile shape' : 'remaining components hold finer variation and noise') + '</span>';
  }

  M.pointer(left.cv, {
    down: function (p) {
      if (!vL.inside(p.x, p.y)) return false;
      kS.set(M.clamp(Math.round(vL.ix(p.x)), 1, D));
    },
    drag: function (p) { kS.set(M.clamp(Math.round(vL.ix(p.x)), 1, D)); },
    hover: function (p) { left.cv.style.cursor = vL.inside(p.x, p.y) ? 'pointer' : ''; }
  });

  load();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
});
