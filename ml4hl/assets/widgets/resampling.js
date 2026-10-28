/* ══════════════════════════════════════════════════════════════════
   Figure 16 — Bootstrap and permutation, built live
   Bootstrap: redraw each group from itself (with replacement) to see how
   much the difference could wobble. Permutation: shuffle the group labels
   to see how big a difference pure chance makes.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('resampling', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Bootstrap and permutation, built live',
    hint: 'Press play to resample &middot; switch between bootstrap and permutation &middot; watch the histogram grow'
  });

  var real = cfg.data && cfg.data.a;
  var seed = cfg.seed || 16, a, b, names, unit, obs, stats, cur = null, rng, TOTAL = 2000;

  function load() {
    if (real) {
      a = cfg.data.a.slice(); b = cfg.data.b.slice();
      names = cfg.data.names || ['group A', 'group B']; unit = cfg.data.unit || '';
    } else {
      var r = M.rng(seed), n = nS.get(), eff = effS.get();
      a = M.range(n).map(function () { return r.normal(0, 1); });
      b = M.range(n).map(function () { return r.normal(eff, 1); });
      names = ['control', 'treatment']; unit = '';
    }
    obs = M.mean(b) - M.mean(a);
    restart();
  }
  function restart() { stats = []; cur = null; rng = M.rng(seed * 3 + 1); }

  function once() {
    var na = a.length, nb = b.length, ra, rb, ia = null, ib = null, lab = null;
    if (modeT.get() === 'bootstrap') {
      ia = M.range(na).map(function () { return rng.int(na); });
      ib = M.range(nb).map(function () { return rng.int(nb); });
      ra = ia.map(function (i) { return a[i]; }); rb = ib.map(function (i) { return b[i]; });
    } else {
      var all = a.concat(b), idx = rng.shuffle(M.range(all.length));
      lab = new Int8Array(all.length);
      idx.forEach(function (j, k) { lab[j] = k < na ? 0 : 1; });
      ra = []; rb = [];
      all.forEach(function (v, j) { (lab[j] ? rb : ra).push(v); });
    }
    var d = M.mean(rb) - M.mean(ra);
    stats.push(d);
    cur = { ia: ia, ib: ib, lab: lab, ma: M.mean(ra), mb: M.mean(rb), d: d };
  }

  /* ─── DOM ─── */
  var modeRow = M.row(root);
  var modeT = M.toggle(modeRow, { label: 'Resampling', value: cfg.mode || 'permutation', options: [['bootstrap', 'bootstrap (how sure?)'], ['permutation', 'permutation (could it be chance?)']], onChange: function () { pause(); restart(); draw(); } });
  var split = M.split(root, 1);
  M.mini(split.a, 'the data (and the current resample)');
  M.mini(split.b, 'difference in means, over many resamples');
  var hf = function (w) { return M.clamp(w * 0.7, 220, 320); };
  var dots = M.stage(split.a, { height: hf, ref: 400, label: 'Two groups of points with their means' });
  var hist = M.stage(split.b, { height: hf, ref: 400, label: 'Histogram of resampled differences in means' });
  var row = M.row(root);
  var playBtn = M.button(row, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Play or pause resampling' });
  M.button(row, 'one resample', function () { pause(); if (stats.length < TOTAL) once(); draw(); }, { aria: 'Draw a single resample' });
  M.button(row, 'all ' + TOTAL, function () { pause(); while (stats.length < TOTAL) once(); draw(); }, { aria: 'Run all resamples at once' });
  var row2 = M.row(root);
  var nS = real ? null : M.slider(row2, { label: 'Points per group', min: 5, max: 80, step: 1, value: cfg.n || 18, accent: 'blue', onChange: function () { pause(); load(); draw(); } });
  var effS = real ? null : M.slider(row2, { label: 'True difference', min: 0, max: 1.5, step: 0.05, value: cfg.effect == null ? 0.6 : cfg.effect, accent: 'orange', onChange: function () { pause(); load(); draw(); } });
  var row3 = M.row(root);
  M.spacer(row3);
  if (!real) M.button(row3, 'new data', function () { pause(); seed++; load(); draw(); }, { aria: 'Draw a new dataset' });
  M.resetButton(row3, function () {
    pause(); seed = cfg.seed || 16;
    modeT.set(cfg.mode || 'permutation', true);
    if (nS) nS.set(cfg.n || 18, true); if (effS) effS.set(cfg.effect == null ? 0.6 : cfg.effect, true);
    load(); draw();
  });
  var read = M.readout(root);

  var acc = 0;
  var loop = M.loop(function (dt) {
    acc += dt;
    var slow = stats.length < 12, wait = slow ? 0.45 : 0;
    if (acc >= wait) {
      acc = 0;
      var k = slow ? 1 : Math.min(TOTAL - stats.length, 6 + Math.floor(stats.length / 40));
      for (var i = 0; i < k; i++) once();
      draw();
    }
    if (stats.length >= TOTAL) { pause(); return false; }
    return true;
  });
  function play() { if (stats.length >= TOTAL) restart(); acc = 1; loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  /* ─── Drawing ─── */
  var vD = M.view(dots, { x: [0, 1], y: [0, 1], pad: { l: 50, r: 12, t: 12, b: 30 } });
  function drawDots() {
    var ctx = dots.ctx, s = dots;
    if (!s.w) return;
    var all = a.concat(b), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), pd = (hi - lo) * 0.08 || 1;
    vD.setDomain([-0.5, 1.5], [lo - pd, hi + pd]);
    M.clear(ctx, s);
    M.axes(ctx, vD, s, { xticks: [0, 1], xfmt: function (v) { return names[v]; }, ylabel: unit ? 'value (' + unit + ')' : 'value', grid: 'y', ny: 4 });
    var jit = M.rng(7), boot = modeT.get() === 'bootstrap';
    [[a, 0, C.blue, cur && cur.ia], [b, 1, C.orange, cur && cur.ib]].forEach(function (g) {
      var vals = g[0], gx = g[1], col = g[2], picks = g[3], count = {};
      if (picks) picks.forEach(function (i) { count[i] = (count[i] || 0) + 1; });
      vals.forEach(function (v, i) {
        var x = vD.px(gx + (jit.next() - 0.5) * 0.42), y = vD.py(v), c = col;
        if (cur && !boot) { var j = gx ? a.length + i : i; c = cur.lab[j] ? C.orange : C.blue; }
        var faded = boot && picks && !count[i];
        M.dot(ctx, x, y, 4 * s.sc, faded ? M.rgba(col, 0.15) : c, '#fff', 1);
        if (boot && count[i] > 1) { ctx.font = M.font(s, 8, 600); ctx.fillStyle = C.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText('\u00d7' + count[i], x + 6 * s.sc, y); }
      });
    });
    var ma = cur ? cur.ma : M.mean(a), mb = cur ? cur.mb : M.mean(b);
    [[0, ma, C.blue], [1, mb, C.orange]].forEach(function (m) {
      ctx.strokeStyle = m[2]; ctx.lineWidth = 2.6 * s.sc;
      ctx.beginPath(); ctx.moveTo(vD.px(m[0] - 0.3), vD.py(m[1])); ctx.lineTo(vD.px(m[0] + 0.3), vD.py(m[1])); ctx.stroke();
    });
    ctx.strokeStyle = M.rgba(C.ink, 0.45); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(vD.px(0.3), vD.py(ma)); ctx.lineTo(vD.px(0.7), vD.py(mb)); ctx.stroke(); ctx.setLineDash([]);
    if (cur && !boot) { ctx.font = M.font(s, 8.5); ctx.fillStyle = C.muted; ctx.textAlign = 'center'; ctx.fillText('labels shuffled: colors no longer mean anything', (vD.L + vD.R) / 2, vD.T + 8 * s.sc); }
  }

  var vH = M.view(hist, { x: [0, 1], y: [0, 1], pad: { l: 14, r: 12, t: 12, b: 38 } });
  function drawHist() {
    var ctx = hist.ctx, s = hist, boot = modeT.get() === 'bootstrap';
    if (!s.w) return;
    var spread = Math.max(Math.abs(obs) * 1.6, (M.std(a) + M.std(b)) * 0.9, 0.2);
    var lo = boot ? obs - spread : -Math.max(spread, Math.abs(obs) * 1.25), hi = boot ? obs + spread : -lo;
    var nb = 40, bins = new Array(nb).fill(0);
    stats.forEach(function (d) { var k = Math.floor((d - lo) / (hi - lo) * nb); if (k >= 0 && k < nb) bins[k]++; });
    var mx = Math.max(4, Math.max.apply(null, bins));
    vH.setDomain([lo, hi], [0, mx * 1.12]);
    M.clear(ctx, s);
    M.axes(ctx, vH, s, { xlabel: 'difference in means' + (unit ? ' (' + unit + ')' : ''), ylabels: false, grid: 'x', nx: 5 });
    var ci = null, p = null;
    if (stats.length >= 50) {
      var srt = stats.slice().sort(function (x, y) { return x - y; });
      if (boot) ci = [M.quantile(srt, 0.025), M.quantile(srt, 0.975)];
      else p = (stats.filter(function (d) { return Math.abs(d) >= Math.abs(obs) - 1e-12; }).length + 1) / (stats.length + 1);
    }
    var bw = (vH.R - vH.L) / nb;
    bins.forEach(function (c, k) {
      var x0 = lo + (k + 0.5) / nb * (hi - lo), shade;
      if (boot) shade = ci && x0 >= ci[0] && x0 <= ci[1] ? C.blue : M.rgba(C.blue, 0.35);
      else shade = Math.abs(x0) >= Math.abs(obs) ? C.red : M.rgba(C.muted, 0.55);
      ctx.fillStyle = shade;
      var y = vH.py(c);
      ctx.fillRect(vH.L + k * bw + 0.5, y, bw - 1, vH.B - y);
    });
    if (ci) {
      ctx.strokeStyle = C.blueDeep; ctx.lineWidth = 3 * s.sc;
      var cy = vH.B - 6 * s.sc;
      ctx.beginPath(); ctx.moveTo(vH.px(ci[0]), cy); ctx.lineTo(vH.px(ci[1]), cy); ctx.stroke();
      if (ci[0] <= 0 && ci[1] >= 0 || lo < 0 && hi > 0) {
        ctx.strokeStyle = M.rgba(C.ink, 0.5); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(vH.px(0), vH.T); ctx.lineTo(vH.px(0), vH.B); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    ctx.strokeStyle = C.red; ctx.lineWidth = 2.2 * s.sc;
    ctx.beginPath(); ctx.moveTo(vH.px(obs), vH.T); ctx.lineTo(vH.px(obs), vH.B); ctx.stroke();
    if (!boot) { ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(vH.px(-obs), vH.T); ctx.lineTo(vH.px(-obs), vH.B); ctx.stroke(); ctx.globalAlpha = 1; }
    M.pill(ctx, s, 'observed ' + M.f(obs, 2), vH.px(obs), vH.T + 10 * s.sc, { align: obs > (lo + hi) / 2 ? 'right' : 'left', color: C.red, size: 9 });
    if (cur) M.dot(ctx, vH.px(cur.d), vH.B - 3 * s.sc, 4 * s.sc, C.ink, '#fff', 1);
    return { ci: ci, p: p };
  }

  function draw() {
    if (!a) return;
    drawDots();
    var r = drawHist(), boot = modeT.get() === 'bootstrap', msg;
    if (!stats.length) msg = '<span class="k">' + (boot ? 'press play to redraw each group from itself, with replacement' : 'press play to shuffle the group labels') + '</span>';
    else if (boot) msg = r.ci ? (r.ci[0] > 0 || r.ci[1] < 0 ? '<span class="g">95% interval ' + M.f(r.ci[0], 2) + ' to ' + M.f(r.ci[1], 2) + ' excludes zero</span>' : '<span class="o">95% interval ' + M.f(r.ci[0], 2) + ' to ' + M.f(r.ci[1], 2) + ' includes zero</span>') : '<span class="k">collecting resamples&hellip;</span>';
    else msg = r.p != null ? (r.p < 0.05 ? '<span class="g">p = ' + M.f(r.p, 3) + ': rare under the exchangeable-label null</span>' : '<span class="o">p = ' + M.f(r.p, 3) + ': compatible with the exchangeable-label null</span>') : '<span class="k">collecting shuffles&hellip;</span>';
    read.innerHTML = '<span class="k">observed difference</span> ' + M.f(obs, 2) + (unit ? ' ' + M.esc(unit) : '') +
      '<span class="sep">&middot;</span><span class="k">resamples</span> ' + stats.length +
      '<span class="sep">&middot;</span>' + msg;
  }

  load();
  dots.draw = draw; hist.draw = draw;
  dots.request();
  return { destroy: function () { loop.stop(); } };
});
