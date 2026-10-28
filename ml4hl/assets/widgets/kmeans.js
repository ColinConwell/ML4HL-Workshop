/* ══════════════════════════════════════════════════════════════════
   Figure 9 — K-means, one step at a time
   Alternate two moves: assign every point to its nearest center, then
   move every center to the middle of its points. Drag a center to start
   over from wherever you like.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('kmeans', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'K-means, one step at a time',
    hint: 'Press play or step &middot; drag a center (the crosses) to restart from there'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 12, initSeed = 1, X, dom, cents, lab, trails, phase, iter, inertias, done, tween = 1, from = null;

  function load() {
    if (real) X = cfg.data.X.map(function (p) { return p.slice(); });
    else {
      var tk = trueS ? trueS.get() : (cfg.true_k || 4);
      var d = M.sim.blobs(M.rng(seed), { k: tk, n: Math.max(6, Math.round((cfg.n || 60) / tk)), spread: 0.24 });
      X = d.X;
    }
    dom = M.bounds(X, 0.1);
    initCenters();
  }

  function initCenters() {
    var k = kS ? kS.get() : (cfg.k || 4), rng = M.rng(seed * 13 + initSeed), how = initT ? initT.get() : (cfg.init || 'random');
    cents = [];
    if (how === 'kmeans++') {
      cents.push(X[rng.int(X.length)].slice());
      while (cents.length < k) {
        var d2 = X.map(function (p) { return Math.min.apply(null, cents.map(function (c) { return M.dist2(p, c); })); });
        var tot = M.sum(d2), u = rng.next() * tot, acc = 0, pick = 0;
        for (var i = 0; i < X.length; i++) { acc += d2[i]; if (acc >= u) { pick = i; break; } }
        cents.push(X[pick].slice());
      }
    } else {
      for (var c = 0; c < k; c++) cents.push([rng.uniform(dom.x[0], dom.x[1]), rng.uniform(dom.y[0], dom.y[1])]);
    }
    restartFrom();
  }
  function restartFrom() {
    lab = new Int16Array(X.length).fill(-1);
    trails = cents.map(function (c) { return [c.slice()]; });
    phase = 'assign'; iter = 0; inertias = []; done = false; tween = 1; from = null;
  }

  function inertia() { var s = 0; for (var i = 0; i < X.length; i++) if (lab[i] >= 0) s += M.dist2(X[i], cents[lab[i]]); return s; }

  /* One half-step: assign, or update. Returns false once nothing changes. */
  function halfStep() {
    if (done) return false;
    if (phase === 'assign') {
      var changed = 0;
      for (var i = 0; i < X.length; i++) {
        var best = 0, bd = Infinity;
        for (var c = 0; c < cents.length; c++) { var d = M.dist2(X[i], cents[c]); if (d < bd) { bd = d; best = c; } }
        if (lab[i] !== best) { changed++; lab[i] = best; }
      }
      inertias.push(inertia());
      if (!changed && iter > 0) { done = true; return false; }
      phase = 'update';
    } else {
      from = cents.map(function (c) { return c.slice(); });
      for (var c2 = 0; c2 < cents.length; c2++) {
        var sx = 0, sy = 0, m = 0;
        for (var j = 0; j < X.length; j++) if (lab[j] === c2) { sx += X[j][0]; sy += X[j][1]; m++; }
        if (m) cents[c2] = [sx / m, sy / m];
        trails[c2].push(cents[c2].slice());
      }
      tween = 0; iter++; phase = 'assign';
      inertias.push(inertia());
    }
    return true;
  }

  /* ─── DOM ─── */
  var stage = M.stage(root, { aspect: 0.54, minH: 260, maxH: 420, label: 'Scatter plot with k-means centers and their territories' });
  var row = M.row(root);
  var playBtn = M.button(row, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Play or pause k-means' });
  M.button(row, 'step', function () { pause(); halfStep(); animate(); }, { aria: 'Take one half-step: assign or update' });
  M.button(row, 'run to the end', function () { pause(); var g = 0; while (halfStep() && g++ < 500); tween = 1; draw(); }, { aria: 'Run until nothing changes' });
  var initT = M.toggle(row, { label: 'Start', value: cfg.init || 'random', options: [['random', 'random'], ['kmeans++', 'k-means++']], onChange: function () { pause(); initCenters(); draw(); } });
  var row2 = M.row(root);
  var kS = M.slider(row2, { label: 'k (centers)', min: 1, max: 8, step: 1, value: cfg.k || 4, accent: 'blue', onInput: function () { pause(); initCenters(); draw(); } });
  var trueS = null;
  if (!real) trueS = M.slider(row2, { label: 'True groups', min: 2, max: 6, step: 1, value: cfg.true_k || 4, accent: 'orange', onInput: function () { pause(); load(); draw(); } });
  var row3 = M.row(root);
  M.button(row3, 'new start', function () { pause(); initSeed++; initCenters(); draw(); }, { aria: 'Re-draw the starting centers' });
  M.spacer(row3);
  if (!real) M.button(row3, 'new data', function () { pause(); seed++; load(); draw(); }, { aria: 'Draw a new random dataset' });
  M.resetButton(row3, function () { pause(); seed = cfg.seed || 12; initSeed = 1; kS.set(cfg.k || 4, true); if (trueS) trueS.set(cfg.true_k || 4, true); initT.set(cfg.init || 'random', true); load(); draw(); });
  var read = M.readout(root);

  var acc = 0;
  var loop = M.loop(function (dt) {
    tween = Math.min(1, tween + dt / 0.45);
    acc += dt;
    if (tween >= 1 && acc > 0.55) {
      acc = 0;
      if (!halfStep()) { pause(); draw(); return false; }
    }
    draw();
    return true;
  });
  var tl = M.loop(function (dt) { tween = Math.min(1, tween + dt / 0.45); draw(); return tween < 1; });
  function animate() { tl.start(); }
  function play() { if (done) restartFrom(); acc = 1; loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  /* ─── Drawing ─── */
  var v = M.view(stage, { x: [0, 1], y: [0, 1], equal: !real, pad: real ? { l: 46, b: 34, t: 12, r: 12 } : { l: 12, r: 12, t: 12, b: 12 } });
  function shown(c) { return from && tween < 1 ? [M.lerp(from[c][0], cents[c][0], M.ease(tween)), M.lerp(from[c][1], cents[c][1], M.ease(tween))] : cents[c]; }

  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w || !X) return;
    v.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    var sh = cents.map(function (c, i) { return shown(i); });
    M.field(ctx, v, 240, 140, function (x, y) {
      var best = 0, bd = Infinity;
      for (var c = 0; c < sh.length; c++) { var dx = x - sh[c][0], dy = y - sh[c][1], d = dx * dx + dy * dy; if (d < bd) { bd = d; best = c; } }
      var col = M.hex(M.catColor(best));
      return [col[0], col[1], col[2], 30];
    });
    if (real) M.axes(ctx, v, s, { grid: false, xlabels: false, ylabels: false, xlabel: cfg.data.xlabel || 'feature 1', ylabel: cfg.data.ylabel || 'feature 2' });
    ctx.save(); v.clip(ctx);
    if (phase === 'update' || tween < 1) {
      ctx.lineWidth = 1;
      for (var i = 0; i < X.length; i++) if (lab[i] >= 0) {
        ctx.strokeStyle = M.rgba(M.catColor(lab[i]), 0.22);
        ctx.beginPath(); ctx.moveTo(v.px(X[i][0]), v.py(X[i][1])); ctx.lineTo(v.px(sh[lab[i]][0]), v.py(sh[lab[i]][1])); ctx.stroke();
      }
    }
    for (i = 0; i < X.length; i++) M.dot(ctx, v.px(X[i][0]), v.py(X[i][1]), 3.4 * s.sc, lab[i] < 0 ? C.muted : M.catColor(lab[i]), '#fff', 0.8);
    trails.forEach(function (t, c) {
      ctx.strokeStyle = M.rgba(M.catColor(c), 0.6); ctx.lineWidth = 1.4 * s.sc; ctx.setLineDash([3 * s.sc, 3 * s.sc]);
      ctx.beginPath();
      t.forEach(function (p, j) { ctx[j ? 'lineTo' : 'moveTo'](v.px(p[0]), v.py(p[1])); });
      ctx.lineTo(v.px(sh[c][0]), v.py(sh[c][1]));
      ctx.stroke(); ctx.setLineDash([]);
    });
    sh.forEach(function (p, c) {
      var px = v.px(p[0]), py = v.py(p[1]), r = 8 * s.sc;
      M.dot(ctx, px, py, r + 2 * s.sc, '#fff');
      ctx.strokeStyle = M.catColor(c); ctx.lineWidth = 3 * s.sc;
      ctx.beginPath(); ctx.moveTo(px - r * 0.7, py - r * 0.7); ctx.lineTo(px + r * 0.7, py + r * 0.7); ctx.moveTo(px + r * 0.7, py - r * 0.7); ctx.lineTo(px - r * 0.7, py + r * 0.7); ctx.stroke();
    });
    ctx.restore();
    drawSpark(ctx, s);
    var next = done ? '<span class="g">converged: no point wants to switch teams</span>'
      : iter === 0 && phase === 'assign' ? '<span class="k">next: give every point to its nearest center</span>'
      : phase === 'assign' ? '<span class="b">next: points re-check which center is nearest</span>'
      : '<span class="o">next: each center moves to the middle of its points</span>';
    var ine = inertias.length ? inertias[inertias.length - 1] : null;
    read.innerHTML = '<span class="k">round</span> ' + iter +
      (ine != null ? '<span class="sep">&middot;</span><span class="k">total squared distance</span> ' + M.f(ine, ine > 100 ? 0 : 2) : '') +
      '<span class="sep">&middot;</span>' + next;
  }

  function drawSpark(ctx, s) {
    if (inertias.length < 2) return;
    var w = 130 * s.sc, h = 44 * s.sc, x0 = v.R - w - 6 * s.sc, y0 = v.T + 6 * s.sc;
    var hi = Math.max.apply(null, inertias), lo = Math.min.apply(null, inertias);
    ctx.save();
    ctx.fillStyle = M.rgba(C.surface, 0.94); ctx.strokeStyle = C.border;
    ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.fill(); ctx.stroke();
    ctx.font = M.font(s, 8.5, 500); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('SPREAD OVER ROUNDS', x0 + 6 * s.sc, y0 + 5 * s.sc);
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.4 * s.sc; ctx.beginPath();
    inertias.forEach(function (e, i) {
      var px = x0 + 6 * s.sc + i / (inertias.length - 1) * (w - 12 * s.sc), py = y0 + h - 6 * s.sc - (e - lo) / ((hi - lo) || 1) * (h - 22 * s.sc);
      ctx[i ? 'lineTo' : 'moveTo'](px, py);
    });
    ctx.stroke(); ctx.restore();
  }

  var drag = -1;
  M.pointer(stage.cv, {
    down: function (p) {
      for (var c = 0; c < cents.length; c++) if (Math.hypot(v.px(cents[c][0]) - p.x, v.py(cents[c][1]) - p.y) < 14 * stage.sc) { drag = c; pause(); stage.cv.style.cursor = 'grabbing'; return; }
      return false;
    },
    drag: function (p) { if (drag < 0) return; cents[drag] = [v.ix(p.x), v.iy(p.y)]; restartFrom(); draw(); },
    up: function () { drag = -1; stage.cv.style.cursor = ''; },
    hover: function (p) {
      var on = cents.some(function (c) { return Math.hypot(v.px(c[0]) - p.x, v.py(c[1]) - p.y) < 14 * stage.sc; });
      stage.cv.style.cursor = on ? 'grab' : '';
    }
  });

  load();
  stage.draw = draw;
  stage.request();
  return { destroy: function () { pause(); tl.stop(); } };
});
