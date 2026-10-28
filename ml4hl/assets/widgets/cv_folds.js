/* ══════════════════════════════════════════════════════════════════
   Figure 15 — K-fold cross-validation (and leakage)
   Every sample takes a turn in the test set. When samples come in groups
   (households, patients, birds), splitting samples instead of groups lets
   groupmates sit on both sides, and the estimate looks better than the
   model really is on new groups.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('cv_folds', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'K-fold cross-validation (and leakage)',
    hint: 'Press play to rotate the folds &middot; switch the splitting scheme &middot; compare with the truth line'
  });

  var seed = cfg.seed || 15, per = 6;
  var S, G, folds, accs, truth;

  /* Groups share a position in feature space AND a private quirk that pushes
     their labels one way. A nearest-neighbor model can memorize the quirk
     only if a groupmate is in the training set. */
  function makeGroups(rng, nG, ge, offset) {
    var out = [];
    for (var g = 0; g < nG; g++) {
      var cx = rng.normal(0, 1), cy = rng.normal(0, 1), quirk = rng.normal(0, 1), sd = 0.75 / (1 + 1.5 * ge);
      for (var i = 0; i < per; i++) {
        var x = [cx + rng.normal(0, sd), cy + rng.normal(0, sd)];
        var y = 1.5 * x[0] + ge * 1.4 * quirk + rng.normal(0, 0.5) > 0 ? 1 : 0;
        out.push({ x: x, y: y, g: g + offset });
      }
    }
    return out;
  }
  function load() {
    var rng = M.rng(seed), nG = gS.get(), ge = geS.get();
    S = makeGroups(rng, nG, ge, 0);
    S = rng.shuffle(S);                          // collection order
    G = nG;
    var fresh = makeGroups(M.rng(seed + 999), 40, ge, 1000);
    truth = M.mean(fresh.map(function (p) { return predict(S, p.x) === p.y ? 1 : 0; }));
    split();
  }

  function predict(train, x) {
    var best = null, bd = Infinity;
    for (var i = 0; i < train.length; i++) { var d = M.dist2(train[i].x, x); if (d < bd) { bd = d; best = train[i]; } }
    return best ? best.y : 0;
  }

  function split() {
    var k = kS.get(), n = S.length, scheme = schemeT.get(), rng = M.rng(seed + 5);
    folds = [];
    var testOf = new Int16Array(n);
    if (scheme === 'kfold') {
      for (var i = 0; i < n; i++) testOf[i] = Math.floor(i * k / n);
      for (var f = 0; f < k; f++) folds.push(M.range(n).filter(function (j) { return testOf[j] === f; }));
    } else if (scheme === 'stratified') {
      var c = [0, 0];
      for (i = 0; i < n; i++) { testOf[i] = c[S[i].y]++ % k; }
      for (f = 0; f < k; f++) folds.push(M.range(n).filter(function (j) { return testOf[j] === f; }));
    } else if (scheme === 'group') {
      var gs = rng.shuffle(M.range(G)), gFold = {};
      gs.forEach(function (g, j) { gFold[g] = j % k; });
      for (f = 0; f < k; f++) folds.push(M.range(n).filter(function (j) { return gFold[S[j].g] === f; }));
    } else {
      var m = Math.max(1, Math.round(n / k));
      for (f = 0; f < k; f++) folds.push(rng.shuffle(M.range(n)).slice(0, m));
    }
    accs = folds.map(function (test) {
      var inTest = {}; test.forEach(function (j) { inTest[j] = 1; });
      var train = S.filter(function (p, j) { return !inTest[j]; });
      return M.mean(test.map(function (j) { return predict(train, S[j].x) === S[j].y ? 1 : 0; }));
    });
    if (tr) tr.setN(k);
  }

  /* ─── DOM ─── */
  var grid = M.stage(root, { height: function (w) { return M.clamp(w * 0.2, 120, 190); }, ref: 760, label: 'Grid of folds: each row is one round, each column one sample, orange cells are the test set' });
  var sp = M.split(root, 1);
  M.mini(sp.a, 'this round: train (filled) vs test (rings)');
  M.mini(sp.b, 'accuracy per round');
  var hf = function (w) { return M.clamp(w * 0.7, 200, 300); };
  var plane = M.stage(sp.a, { height: hf, ref: 400, label: 'Scatter of samples for the current fold' });
  var bars = M.stage(sp.b, { height: hf, ref: 400, label: 'Bar chart of per-fold accuracy with the true accuracy line' });
  var tr = M.transport(root, { n: cfg.k || 5, fps: 1.4, loop: false, fmt: function (i, nn) { return 'round ' + (i + 1) + ' / ' + nn; }, onFrame: function () { draw(); } });
  var row = M.row(root);
  var schemeT = M.toggle(row, {
    label: 'Split by', value: cfg.scheme || 'kfold',
    options: [['kfold', 'k-fold'], ['stratified', 'stratified'], ['shuffle', 'shuffle-split'], ['group', 'group k-fold']],
    onChange: function () { split(); tr.show(0); draw(); }
  });
  var row2 = M.row(root);
  var kS = M.slider(row2, { label: 'Folds (k)', min: 2, max: 10, step: 1, value: cfg.k || 5, accent: 'blue', onChange: function () { split(); tr.show(0); draw(); } });
  var gS = M.slider(row2, { label: 'Groups', min: 4, max: 20, step: 1, value: cfg.groups || 12, accent: 'blue', onChange: function () { load(); tr.show(0); draw(); } });
  var row3 = M.row(root);
  var geS = M.slider(row3, { label: 'How alike groupmates are', min: 0, max: 2, step: 0.05, value: cfg.group_effect == null ? 1.2 : cfg.group_effect, accent: 'orange', onChange: function () { load(); draw(); } });
  M.spacer(row3);
  M.button(row3, 'new data', function () { seed++; load(); tr.show(0); draw(); }, { aria: 'Draw a new dataset' });
  M.resetButton(row3, function () {
    seed = cfg.seed || 15; tr.pause();
    kS.set(cfg.k || 5, true); gS.set(cfg.groups || 12, true); geS.set(cfg.group_effect == null ? 1.2 : cfg.group_effect, true);
    schemeT.set(cfg.scheme || 'kfold', true);
    load(); tr.show(0); draw();
  });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  function groupCol(g) { return M.catColor(g % 12); }

  function drawGrid() {
    var ctx = grid.ctx, s = grid;
    if (!s.w) return;
    M.clear(ctx, s);
    var n = S.length, k = folds.length, lm = 64 * s.sc, top = 30 * s.sc, bot = 6 * s.sc;
    var cw = (s.w - lm - 10 * s.sc) / n, rh = Math.min(26 * s.sc, (s.h - top - bot) / k);
    var disp = M.range(n).sort(function (a, b) { return S[a].g - S[b].g || a - b; });
    var col = {}; disp.forEach(function (j, c) { col[j] = c; });
    ctx.font = M.font(s, 8.5, 500); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = C.muted;
    ctx.fillText('GROUP', lm - 6 * s.sc, 9 * s.sc);
    ctx.fillText('LABEL', lm - 6 * s.sc, 20 * s.sc);
    disp.forEach(function (j, c) {
      ctx.fillStyle = groupCol(S[j].g); ctx.fillRect(lm + c * cw, 4 * s.sc, Math.ceil(cw), 9 * s.sc);
      ctx.fillStyle = S[j].y ? C.ink : '#ddd8cc'; ctx.fillRect(lm + c * cw, 16 * s.sc, Math.ceil(cw), 7 * s.sc);
    });
    for (var f = 0; f < k; f++) {
      var y = top + f * rh, inTest = {};
      folds[f].forEach(function (j) { inTest[j] = 1; });
      var on = f === tr.i;
      ctx.globalAlpha = on ? 1 : 0.45;
      disp.forEach(function (j, c) {
        ctx.fillStyle = inTest[j] ? C.orange : C.blueSoft;
        ctx.fillRect(lm + c * cw + 0.5, y + 1, Math.max(1, cw - 1), rh - 2);
      });
      ctx.globalAlpha = 1;
      ctx.fillStyle = on ? C.ink : C.muted; ctx.font = M.font(s, 8.5, on ? 600 : 400);
      ctx.fillText('round ' + (f + 1), lm - 6 * s.sc, y + rh / 2);
    }
  }

  var vP = M.view(plane, { x: [-1, 1], y: [-1, 1], equal: true, pad: { l: 10, r: 10, t: 10, b: 10 } });
  function drawPlane() {
    var ctx = plane.ctx, s = plane;
    if (!s.w) return;
    var bb = M.bounds(S.map(function (p) { return p.x; }), 0.1);
    vP.setDomain(bb.x, bb.y);
    M.clear(ctx, s);
    var test = {}; (folds[tr.i] || []).forEach(function (j) { test[j] = 1; });
    var train = S.filter(function (p, j) { return !test[j]; });
    ctx.save(); vP.clip(ctx);
    S.forEach(function (p, j) {
      var col = p.y ? C.ink : C.faint;
      if (test[j]) {
        var nb = null, bd = Infinity;
        train.forEach(function (q) { var d = M.dist2(q.x, p.x); if (d < bd) { bd = d; nb = q; } });
        if (nb) {
          ctx.strokeStyle = nb.g === p.g ? M.rgba(C.red, 0.7) : M.rgba(C.muted, 0.5); ctx.lineWidth = 1.2 * s.sc;
          ctx.beginPath(); ctx.moveTo(vP.px(p.x[0]), vP.py(p.x[1])); ctx.lineTo(vP.px(nb.x[0]), vP.py(nb.x[1])); ctx.stroke();
        }
      }
    });
    S.forEach(function (p, j) {
      if (test[j]) M.dot(ctx, vP.px(p.x[0]), vP.py(p.x[1]), 5 * s.sc, '#fff', C.orange, 2 * s.sc);
      else M.dot(ctx, vP.px(p.x[0]), vP.py(p.x[1]), 3.4 * s.sc, groupCol(p.g), null);
    });
    S.forEach(function (p, j) { if (test[j]) M.dot(ctx, vP.px(p.x[0]), vP.py(p.x[1]), 2 * s.sc, p.y ? C.ink : C.faint, null); });
    ctx.restore();
    ctx.font = M.font(s, 8.5); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('red link = nearest training point is a groupmate', vP.L + 4 * s.sc, vP.B - 3 * s.sc);
  }

  var vB = M.view(bars, { x: [0, 1], y: [0, 1], pad: { l: 44, r: 12, t: 14, b: 30 } });
  function drawBars() {
    var ctx = bars.ctx, s = bars, k = accs.length;
    if (!s.w) return;
    vB.setDomain([0, k], [0.3, 1]);
    M.clear(ctx, s);
    M.axes(ctx, vB, s, { xlabels: false, ylabel: 'accuracy', yticks: [0.4, 0.6, 0.8, 1], yfmt: function (v) { return Math.round(v * 100) + '%'; }, grid: 'y' });
    var shown = accs.slice(0, tr.i + 1), bw = (vB.R - vB.L) / k;
    accs.forEach(function (a, f) {
      var x = vB.L + f * bw + bw * 0.18, w = bw * 0.64, on = f <= tr.i;
      ctx.fillStyle = on ? (f === tr.i ? C.orange : M.rgba(C.orange, 0.55)) : M.rgba(C.faint, 0.35);
      var y = vB.py(on ? a : 0.3);
      ctx.fillRect(x, y, w, vB.B - y);
      ctx.font = M.font(s, 8.5); ctx.fillStyle = C.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(String(f + 1), x + w / 2, vB.B + 4 * s.sc);
    });
    var mean = M.mean(shown);
    ctx.strokeStyle = C.orangeDeep; ctx.lineWidth = 2 * s.sc;
    ctx.beginPath(); ctx.moveTo(vB.L, vB.py(mean)); ctx.lineTo(vB.R, vB.py(mean)); ctx.stroke();
    ctx.strokeStyle = C.green; ctx.setLineDash([6 * s.sc, 4 * s.sc]);
    ctx.beginPath(); ctx.moveTo(vB.L, vB.py(truth)); ctx.lineTo(vB.R, vB.py(truth)); ctx.stroke(); ctx.setLineDash([]);
    M.pill(ctx, s, 'CV estimate ' + M.pct(mean), vB.R - 2 * s.sc, vB.py(mean) - 11 * s.sc, { align: 'right', color: C.orangeDeep, size: 9 });
    M.pill(ctx, s, 'truth on new groups ' + M.pct(truth), vB.R - 2 * s.sc, vB.py(truth) + 11 * s.sc, { align: 'right', color: C.green, size: 9 });
  }

  function draw() {
    if (!folds) return;
    drawGrid(); drawPlane(); drawBars();
    var shown = accs.slice(0, tr.i + 1), mean = M.mean(shown), gap = mean - truth, scheme = schemeT.get();
    var verdict = gap > 0.08 ? '<span class="r">leakage: groupmates on both sides make the estimate ' + M.pct(gap) + ' too rosy</span>'
      : Math.abs(gap) <= 0.08 ? '<span class="g">honest: the estimate matches performance on new groups</span>'
      : '<span class="b">a bit pessimistic, which is the safe direction</span>';
    read.innerHTML = '<span class="k">' + S.length + ' samples in ' + G + ' groups</span>' +
      '<span class="sep">&middot;</span><span class="k">' + ({ kfold: 'k-fold', stratified: 'stratified k-fold', shuffle: 'shuffle-split', group: 'group k-fold' })[scheme] + ' estimate</span> ' + M.pct(mean) +
      '<span class="sep">&middot;</span><span class="k">truth</span> ' + M.pct(truth) +
      '<span class="sep">&middot;</span>' + verdict;
  }

  load();
  grid.draw = draw; plane.draw = draw; bars.draw = draw;
  tr.show(0);
  grid.request();
  return { destroy: function () { tr.pause(); } };
});
