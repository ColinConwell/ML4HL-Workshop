/* ══════════════════════════════════════════════════════════════════
   Figure 13 — Building and comparing RDMs
   Two systems describe the same items with different features. Turn each
   into a table of pairwise dissimilarities (an RDM), then compare the
   tables: that second-order comparison is representational similarity
   analysis.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('rdm_builder', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Building and comparing RDMs',
    hint: 'Hover a cell &middot; switch the distance &middot; shuffle the order &middot; run the shuffle test'
  });

  var real = cfg.data && cfg.data.A;
  var seed = cfg.seed || 10;
  var names, groups, A, B, labelA, labelB, n, order, rdmA, rdmB, rankA, rankB, rho, perm = null, hover = null;

  var SIM = {
    animal: ['cat', 'dog', 'horse', 'owl', 'frog', 'shark'],
    tool: ['hammer', 'saw', 'wrench', 'drill', 'pliers', 'shovel'],
    food: ['apple', 'bread', 'cheese', 'carrot', 'grape', 'pizza'],
    vehicle: ['car', 'bus', 'bike', 'boat', 'plane', 'train']
  };

  function load() {
    if (real) {
      names = cfg.data.names; A = cfg.data.A; B = cfg.data.B;
      groups = cfg.data.groups || names.map(function () { return 'item'; });
      labelA = cfg.data.labelA || 'system A'; labelB = cfg.data.labelB || 'system B';
    } else {
      var rng = M.rng(seed), noise = noiseS ? noiseS.get() : (cfg.noise_b == null ? 0.8 : cfg.noise_b);
      names = []; groups = []; A = [];
      Object.keys(SIM).forEach(function (g) {
        var cen = M.range(8).map(function () { return rng.normal(0, 1); });
        SIM[g].forEach(function (nm) { names.push(nm); groups.push(g); A.push(cen.map(function (c) { return c + rng.normal(0, 0.55); })); });
      });
      var W = M.range(12).map(function () { return M.range(8).map(function () { return rng.normal(0, 0.45); }); });
      var rngB = M.rng(seed + 500);
      B = A.map(function (a) { return W.map(function (w) { var s = 0; for (var j = 0; j < 8; j++) s += w[j] * a[j]; return Math.tanh(s) + rngB.normal(0, noise); }); });
      labelA = 'brain region (8 features)'; labelB = 'model layer (12 features)';
    }
    n = names.length;
    setOrder();
  }
  var gNames = [];
  function setOrder() {
    gNames = [];
    groups.forEach(function (g) { if (gNames.indexOf(g) < 0) gNames.push(g); });
    order = M.range(n);
    if (orderT && orderT.get() === 'shuffled') order = M.rng(seed + 9).shuffle(order);
    else order.sort(function (a, b) { return gNames.indexOf(groups[a]) - gNames.indexOf(groups[b]) || a - b; });
  }

  function dissim(x, y, metric) {
    if (metric === 'euclidean') return M.dist(x, y);
    if (metric === 'cosine') { var d = 0, nx = 0, ny = 0; for (var i = 0; i < x.length; i++) { d += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i]; } return 1 - d / Math.sqrt(nx * ny || 1); }
    return 1 - M.pearson(x, y);
  }
  function build() {
    var metric = metricT.get();
    rdmA = []; rdmB = [];
    for (var i = 0; i < n; i++) { rdmA.push(new Float64Array(n)); rdmB.push(new Float64Array(n)); }
    var ua = [], ub = [];
    for (i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
      rdmA[i][j] = rdmA[j][i] = dissim(A[i], A[j], metric);
      rdmB[i][j] = rdmB[j][i] = dissim(B[i], B[j], metric);
      ua.push(rdmA[i][j]); ub.push(rdmB[i][j]);
    }
    var ra = M.rank(ua), rb = M.rank(ub), k = 0, m = ua.length;
    rankA = []; rankB = [];
    for (i = 0; i < n; i++) { rankA.push(new Float64Array(n)); rankB.push(new Float64Array(n)); }
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) { rankA[i][j] = rankA[j][i] = (ra[k] - 1) / Math.max(1, m - 1); rankB[i][j] = rankB[j][i] = (rb[k] - 1) / Math.max(1, m - 1); k++; }
    rho = M.pearson(ra, rb);
    perm = null;
  }

  function permTest() {
    var rng = M.rng(seed + 321), nulls = [], ua = [];
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) ua.push(rankA[i][j]);
    for (var t = 0; t < 500; t++) {
      var p = rng.shuffle(M.range(n)), ub = [];
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) ub.push(rankB[p[i]][p[j]]);
      nulls.push(M.pearson(ua, ub));
    }
    var beat = nulls.filter(function (v) { return v >= rho; }).length;
    perm = { nulls: nulls, p: (beat + 1) / (nulls.length + 1) };
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1.7);
  M.mini(split.a, 'two dissimilarity tables (light = similar, dark = different)');
  M.mini(split.b, 'every pair: table A vs table B');
  var hf = function (w) { return M.clamp(w * 0.56, 230, 360); };
  var mats = M.stage(split.a, { height: hf, ref: 560, label: 'Two representational dissimilarity matrices side by side' });
  var sc = M.stage(split.b, { height: function (w) { return M.clamp(w * 0.95, 230, 360); }, ref: 320, label: 'Scatter of dissimilarities in system A versus system B' });
  var row = M.row(root);
  var metricT = M.toggle(row, { label: 'Distance', value: cfg.metric || 'correlation', options: [['euclidean', 'euclidean'], ['correlation', 'correlation'], ['cosine', 'cosine']], onChange: function () { build(); draw(); } });
  var orderT = M.toggle(row, { label: 'Order', value: cfg.order || 'category', options: [['category', 'by category'], ['shuffled', 'shuffled']], onChange: function () { setOrder(); draw(); } });
  var row2 = M.row(root);
  var noiseS = real ? null : M.slider(row2, { label: 'How different B is', min: 0, max: 2.5, step: 0.05, value: cfg.noise_b == null ? 0.8 : cfg.noise_b, accent: 'orange', onInput: function () { load(); build(); draw(); } });
  M.button(row2, 'shuffle test', function () { permTest(); draw(); }, { aria: 'Run a 500-shuffle permutation test' });
  M.spacer(row2);
  if (!real) M.button(row2, 'new data', function () { seed++; load(); build(); draw(); }, { aria: 'Draw new random systems' });
  M.resetButton(row2, function () {
    seed = cfg.seed || 10; hover = null;
    metricT.set(cfg.metric || 'correlation', true); orderT.set(cfg.order || 'category', true);
    if (noiseS) noiseS.set(cfg.noise_b == null ? 0.8 : cfg.noise_b, true);
    load(); build(); draw();
  });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var geom = null;
  function groupColor(g) { return M.catColor(Math.max(0, gNames.indexOf(g))); }

  function drawMats() {
    var ctx = mats.ctx, s = mats;
    if (!s.w) return;
    M.clear(ctx, s);
    var showNames = n <= 30;
    var lm = showNames ? 52 * s.sc : 10 * s.sc, gap = 18 * s.sc, top = 22 * s.sc, bar = 5 * s.sc;
    var hs = Math.min((s.w - lm - gap - 2 * bar - 10 * s.sc) / 2, s.h - top - 12 * s.sc), cell = hs / n;
    var x0s = [lm + bar + 2, lm + bar + 2 + hs + gap + bar];
    geom = { x0s: x0s, top: top, cell: cell, hs: hs };
    [[rankA, labelA], [rankB, labelB]].forEach(function (pair, m) {
      var R = pair[0], x0 = x0s[m];
      for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) {
        var i = order[a], j = order[b];
        ctx.fillStyle = i === j ? '#ffffff' : M.css(M.seq(0.08 + 0.92 * R[i][j]));
        ctx.fillRect(x0 + b * cell, top + a * cell, Math.ceil(cell), Math.ceil(cell));
      }
      for (a = 0; a < n; a++) {
        ctx.fillStyle = groupColor(groups[order[a]]);
        ctx.fillRect(x0 - bar - 2, top + a * cell, bar, Math.ceil(cell));
        ctx.fillRect(x0 + a * cell, top - bar - 2, Math.ceil(cell), bar);
      }
      ctx.font = M.font(s, 9, 600); ctx.fillStyle = m ? C.orange : C.blue; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(M.strip(pair[1]).toUpperCase().slice(0, 34), x0 - bar - 2, top - bar - 5);
      if (hover) {
        var ha = order.indexOf(hover[0]), hb = order.indexOf(hover[1]);
        ctx.strokeStyle = C.red; ctx.lineWidth = 2;
        ctx.strokeRect(x0 + hb * cell, top + ha * cell, cell, cell);
        ctx.strokeRect(x0 + ha * cell, top + hb * cell, cell, cell);
      }
    });
    if (showNames) {
      ctx.font = M.font(s, Math.min(9, cell * 0.85)); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (var a = 0; a < n; a++) {
        var i = order[a], on = hover && (hover[0] === i || hover[1] === i);
        ctx.fillStyle = on ? C.red : C.muted;
        ctx.fillText(String(names[i]).slice(0, 11), lm - 2, top + (a + 0.5) * cell);
      }
    }
  }

  var vS = M.view(sc, { x: [0, 1], y: [0, 1], pad: { l: 46, r: 10, t: 12, b: 38 } });
  function drawScatter() {
    var ctx = sc.ctx, s = sc;
    if (!s.w) return;
    var xs = [], ys = [], cs = [], ij = [];
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
      xs.push(rdmA[i][j]); ys.push(rdmB[i][j]); ij.push([i, j]);
      cs.push(groups[i] === groups[j] ? groupColor(groups[i]) : null);
    }
    var bx = [Math.min.apply(null, xs), Math.max.apply(null, xs)], by = [Math.min.apply(null, ys), Math.max.apply(null, ys)];
    var px = (bx[1] - bx[0]) * 0.06 || 0.1, py = (by[1] - by[0]) * 0.06 || 0.1;
    vS.setDomain([bx[0] - px, bx[1] + px], [by[0] - py, by[1] + py]);
    M.clear(ctx, s);
    M.axes(ctx, vS, s, { xlabel: 'A: dissimilarity', ylabel: 'B: dissimilarity', nx: 3, ny: 4 });
    ctx.save(); vS.clip(ctx);
    for (i = 0; i < xs.length; i++) if (!cs[i]) M.dot(ctx, vS.px(xs[i]), vS.py(ys[i]), 2.2 * s.sc, M.rgba(C.muted, 0.45), null);
    for (i = 0; i < xs.length; i++) if (cs[i]) M.dot(ctx, vS.px(xs[i]), vS.py(ys[i]), 2.8 * s.sc, cs[i], null);
    if (hover) {
      var a = Math.min(hover[0], hover[1]), b = Math.max(hover[0], hover[1]);
      M.dot(ctx, vS.px(rdmA[a][b]), vS.py(rdmB[a][b]), 6 * s.sc, null, C.red, 2 * s.sc);
    }
    ctx.restore();
    M.pill(ctx, s, 'rank agreement \u03c1 = ' + M.f(rho, 2), vS.L + 6 * s.sc, vS.T + 11 * s.sc, { color: C.ink, size: 9.5 });
    if (perm) drawPerm(ctx, s);
  }
  function drawPerm(ctx, s) {
    var w = 118 * s.sc, h = 52 * s.sc, x0 = vS.R - w - 4 * s.sc, y0 = vS.B - h - 4 * s.sc;
    var lo = Math.min(Math.min.apply(null, perm.nulls), -0.2), hi = Math.max(rho, Math.max.apply(null, perm.nulls)) + 0.05, nb = 24, bins = new Array(nb).fill(0);
    perm.nulls.forEach(function (v) { bins[Math.min(nb - 1, Math.floor((v - lo) / (hi - lo) * nb))]++; });
    var mx = Math.max.apply(null, bins);
    ctx.fillStyle = M.rgba(C.surface, 0.95); ctx.strokeStyle = C.border;
    ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.fill(); ctx.stroke();
    ctx.font = M.font(s, 8, 500); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('SHUFFLED \u03c1 (500x)', x0 + 5 * s.sc, y0 + 4 * s.sc);
    var bw = (w - 10 * s.sc) / nb, base = y0 + h - 5 * s.sc, hh = h - 22 * s.sc;
    ctx.fillStyle = M.rgba(C.muted, 0.6);
    bins.forEach(function (c, i) { var bh = c / mx * hh; ctx.fillRect(x0 + 5 * s.sc + i * bw, base - bh, bw - 1, bh); });
    var rx = x0 + 5 * s.sc + (rho - lo) / (hi - lo) * (w - 10 * s.sc);
    ctx.strokeStyle = C.red; ctx.lineWidth = 2 * s.sc; ctx.beginPath(); ctx.moveTo(rx, base); ctx.lineTo(rx, base - hh); ctx.stroke();
  }

  function draw() {
    if (!rdmA) return;
    drawMats(); drawScatter();
    var strength = rho > 0.6 ? '<span class="g">the two systems carve up the items the same way</span>'
      : rho > 0.25 ? '<span class="b">partly shared structure</span>' : '<span class="o">little shared structure</span>';
    read.innerHTML = '<span class="k">' + n + ' items, ' + (n * (n - 1) / 2) + ' pairs</span>' +
      '<span class="sep">&middot;</span><span class="k">rank agreement</span> \u03c1 = ' + M.f(rho, 2) +
      (perm ? '<span class="sep">&middot;</span><span class="k">shuffle test</span> p ' + (perm.p < 0.002 ? '&lt; 0.002' : '= ' + M.f(perm.p, 3)) : '') +
      (hover ? '<span class="sep">&middot;</span><span class="r">' + M.esc(names[hover[0]]) + ' vs ' + M.esc(names[hover[1]]) + '</span>' : '') +
      '<span class="sep">&middot;</span>' + strength;
  }

  M.pointer(mats.cv, {
    down: function () { return false; },
    hover: function (p) {
      if (!geom) return;
      var h = null;
      geom.x0s.forEach(function (x0) {
        var b = Math.floor((p.x - x0) / geom.cell), a = Math.floor((p.y - geom.top) / geom.cell);
        if (a >= 0 && b >= 0 && a < n && b < n && a !== b) h = [order[a], order[b]];
      });
      if (String(h) !== String(hover)) { hover = h; draw(); }
    },
    leave: function () { if (hover) { hover = null; draw(); } }
  });

  load();
  build();
  mats.draw = draw; sc.draw = draw;
  mats.request();
});
