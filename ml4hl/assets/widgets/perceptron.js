/* ══════════════════════════════════════════════════════════════════
   Figure 6 — A linear perceptron, live
   Left: the network (inputs × weights → sum → threshold → output).
   Right: the same weights drawn as a boundary in the input plane.
   The perceptron rule nudges the weights after every mistake.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('perceptron', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'A linear perceptron, live',
    hint: 'Press play or step &middot; click the plane to add points (shift-click for the other class)'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 2, LR = cfg.lr || 0.25;
  var X, y, w, order, k, epoch, mistakes, lastMistakes, status, cur = null, phase = 0, flash = 0;

  function makePreset(name) {
    var rng = M.rng(seed), n = cfg.n || 24, P = [], Y = [];
    if (name === 'xor') {
      [[-0.6, -0.6, 0], [0.6, 0.6, 0], [-0.6, 0.6, 1], [0.6, -0.6, 1]].forEach(function (c) {
        for (var i = 0; i < n / 4; i++) { P.push([c[0] + rng.normal(0, 0.14), c[1] + rng.normal(0, 0.14)]); Y.push(c[2]); }
      });
    } else {
      var a = rng.uniform(0, Math.PI * 2);
      for (var c = 0; c < 2; c++) {
        var s = c ? 1 : -1;
        for (var i = 0; i < n / 2; i++) { P.push([s * 0.5 * Math.cos(a) + rng.normal(0, 0.2), s * 0.5 * Math.sin(a) + 0.1 + rng.normal(0, 0.2)]); Y.push(c); }
      }
    }
    X = P; y = Y;
  }
  function resetWeights() {
    var rng = M.rng(seed * 7 + 1);
    w = [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3)];
    epoch = 0; mistakes = 0; lastMistakes = null; status = 'ready'; cur = null; phase = 0;
    newOrder();
  }
  function newOrder() { order = M.rng(seed + epoch * 31).shuffle(M.range(X.length)); k = 0; }
  function load() {
    if (real) { X = cfg.data.X.map(function (p) { return p.slice(); }); y = cfg.data.y.slice(); }
    else makePreset(presetT ? presetT.get() : (cfg.preset || 'separable'));
    resetWeights();
  }

  function act(p) { return w[0] * p[0] + w[1] * p[1] + w[2]; }
  function pred(p) { return act(p) >= 0 ? 1 : 0; }

  /* Present the next example; the update itself is applied at the end of the animation. */
  function present() {
    if (!X.length) return false;
    if (k >= order.length) {
      epoch++; lastMistakes = mistakes;
      if (mistakes === 0) { status = 'converged'; return false; }
      if (epoch >= 25) { status = 'stuck'; return false; }
      mistakes = 0; newOrder();
    }
    var i = order[k++], p = X[i], s = act(p), yhat = s >= 0 ? 1 : 0;
    cur = { i: i, p: p, s: s, yhat: yhat, err: y[i] - yhat };
    if (cur.err) mistakes++;
    status = 'training';
    return true;
  }
  function applyUpdate() {
    if (!cur || !cur.err) return;
    w[0] += LR * cur.err * cur.p[0]; w[1] += LR * cur.err * cur.p[1]; w[2] += LR * cur.err;
    flash = 1;
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, 'the network');
  M.mini(split.b, 'what the network sees');
  var hf = function (wd) { return M.clamp(wd * 0.86, 220, 340); };
  var net = M.stage(split.a, { height: hf, ref: 400, label: 'Perceptron diagram with weighted connections' });
  var plane = M.stage(split.b, { height: hf, ref: 400, label: 'Input plane with the perceptron decision boundary' });
  var legend = M.legend(root, [
    { color: C.blue, label: real ? (cfg.data.names || ['class 0'])[0] : 'class A (output 0)' },
    { color: C.orange, label: real ? (cfg.data.names || ['', 'class 1'])[1] : 'class B (output 1)' },
    { color: C.ink, label: 'boundary: weighted sum = 0', line: true },
    { color: C.green, label: 'positive weight', line: true }, { color: C.red, label: 'negative weight', line: true }
  ]);
  var row = M.row(root);
  var playBtn = M.button(row, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Play or pause training' });
  M.button(row, 'step', function () { pause(); finishCurrent(); if (present()) { phase = 0; stepLoop.start(); } drawAll(); }, { aria: 'Present one example' });
  M.button(row, 'run epoch', function () { pause(); fastEpoch(); }, { aria: 'Run one full pass over the data instantly' });
  var presetT = null;
  if (!real) presetT = M.toggle(row, { label: 'Data', value: cfg.preset || 'separable', options: [['separable', 'separable'], ['xor', 'XOR']], onChange: function () { pause(); load(); drawAll(); } });
  var row2 = M.row(root);
  M.slider(row2, { label: 'Nudge size', min: 0.02, max: 1, step: 0.01, value: LR, accent: 'green', fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { LR = v; } });
  M.spacer(row2);
  M.button(row2, 'clear points', function () { pause(); X = []; y = []; resetWeights(); drawAll(); }, { aria: 'Remove every point' });
  M.resetButton(row2, function () { pause(); load(); drawAll(); });
  var read = M.readout(root);

  var SPEED = 0.75, acc = 0;
  function finishCurrent() { if (cur && phase < 1) { applyUpdate(); phase = 1; } }
  var loop = M.loop(function (dt) {
    phase += dt / SPEED; flash = Math.max(0, flash - dt * 2.5);
    if (phase >= 1) {
      if (cur && !cur.done) { applyUpdate(); cur.done = true; }
      acc += dt;
      if (acc > 0.12) {
        acc = 0;
        if (!present()) { pause(); drawAll(); return false; }
        phase = 0;
      }
    }
    drawAll();
    return true;
  });
  var stepLoop = M.loop(function (dt) {
    phase = Math.min(1, phase + dt / SPEED); flash = Math.max(0, flash - dt * 2.5);
    if (phase >= 1 && cur && !cur.done) { applyUpdate(); cur.done = true; }
    drawAll();
    return phase < 1 || flash > 0;
  });
  function play() {
    if (status === 'converged' || status === 'stuck') { resetWeights(); }
    stepLoop.stop();
    if (!cur || cur.done) { if (!present()) return; phase = 0; }
    loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on');
  }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }
  function fastEpoch() {
    finishCurrent();
    var e0 = epoch, guard = 0;
    while (epoch === e0 && guard++ < 10000) {
      if (!present()) break;
      applyUpdate(); cur.done = true;
    }
    phase = 1; flash = 0;
    drawAll();
  }

  /* ─── Drawing: network ─── */
  function drawNet() {
    var ctx = net.ctx, s = net;
    if (!s.w) return;
    M.clear(ctx, s);
    var W = s.w, H = s.h, sc = s.sc;
    var inX = W * 0.13, sumX = W * 0.56, stepX = W * 0.76, outX = W * 0.92, midY = H * 0.5;
    var ins = [
      { label: 'x\u2081', y: H * 0.2, val: cur ? cur.p[0] : null, w: w[0] },
      { label: 'x\u2082', y: H * 0.5, val: cur ? cur.p[1] : null, w: w[1] },
      { label: '1', y: H * 0.8, val: 1, w: w[2], bias: true }
    ];
    var maxW = Math.max(0.5, Math.abs(w[0]), Math.abs(w[1]), Math.abs(w[2]));
    var fwd = M.clamp(phase / 0.55, 0, 1), back = cur && cur.err ? M.clamp((phase - 0.62) / 0.38, 0, 1) : 0;

    ins.forEach(function (n, idx) {
      var col = n.w >= 0 ? C.green : C.red;
      ctx.strokeStyle = M.rgba(col, 0.25 + 0.65 * Math.abs(n.w) / maxW);
      ctx.lineWidth = (1 + 6 * Math.abs(n.w) / maxW) * sc;
      ctx.beginPath(); ctx.moveTo(inX, n.y); ctx.lineTo(sumX, midY); ctx.stroke();
      var mx = M.lerp(inX, sumX, 0.42), my = M.lerp(n.y, midY, 0.42);
      M.pill(ctx, s, (n.bias ? 'bias ' : 'w' + (idx + 1) + ' ') + M.f(n.w, 2), mx, my - 12 * sc, { align: 'center', color: col, size: 9 });
      if (cur && fwd > 0 && fwd < 1) M.dot(ctx, M.lerp(inX, sumX, fwd), M.lerp(n.y, midY, fwd), 4 * sc, C.ink);
      if (back > 0 && back < 1) M.dot(ctx, M.lerp(sumX, inX, back), M.lerp(midY, n.y, back), 4.5 * sc, C.red);
      M.dot(ctx, inX, n.y, 16 * sc, C.surface, C.ink, 1.4 * sc);
      ctx.font = M.font(s, 12, 500, 'sans'); ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(n.label, inX, n.y);
      if (n.val != null && !n.bias) {
        ctx.font = M.font(s, 9); ctx.fillStyle = C.muted; ctx.textAlign = 'right';
        ctx.fillText(M.f(n.val, 2), inX - 21 * sc, n.y);
      }
    });
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6 * sc;
    ctx.beginPath(); ctx.moveTo(sumX + 18 * sc, midY); ctx.lineTo(stepX - 18 * sc, midY); ctx.lineTo(outX - 16 * sc, midY); ctx.stroke();
    M.dot(ctx, sumX, midY, 18 * sc, C.surface, C.ink, 1.6 * sc);
    ctx.font = M.font(s, 15, 600, 'sans'); ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('\u03a3', sumX, midY + 1);
    if (cur && fwd >= 1) {
      ctx.font = M.font(s, 9); ctx.fillStyle = C.muted; ctx.fillText('sum ' + M.f(cur.s, 2), sumX, midY + 30 * sc);
    }
    var bw = 26 * sc;
    ctx.fillStyle = C.surface; ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4 * sc;
    ctx.beginPath(); ctx.rect(stepX - bw / 2, midY - bw / 2, bw, bw); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(stepX - bw * 0.35, midY + bw * 0.25); ctx.lineTo(stepX, midY + bw * 0.25); ctx.lineTo(stepX, midY - bw * 0.25); ctx.lineTo(stepX + bw * 0.35, midY - bw * 0.25);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8 * sc; ctx.stroke();
    ctx.font = M.font(s, 8.5, 500); ctx.fillStyle = C.muted; ctx.fillText('THRESHOLD', stepX, midY - bw / 2 - 9 * sc);
    var showOut = cur && phase >= 0.55;
    var outCol = showOut ? (cur.yhat ? C.orange : C.blue) : C.surface;
    M.dot(ctx, outX, midY, 15 * sc, outCol, showOut && cur.err ? C.red : C.ink, (showOut && cur.err ? 3 : 1.4) * sc);
    ctx.font = M.font(s, 12, 600, 'sans'); ctx.fillStyle = showOut ? C.surface : C.ink; ctx.fillText(showOut ? String(cur.yhat) : 'y', outX, midY + 1);
    if (showOut) {
      ctx.font = M.font(s, 9, 500); ctx.fillStyle = cur.err ? C.red : C.green; ctx.textAlign = 'center';
      ctx.fillText(cur.err ? 'WRONG: NUDGE' : 'RIGHT', outX - 4 * sc, midY + 30 * sc);
      ctx.fillStyle = C.muted; ctx.font = M.font(s, 9);
      ctx.fillText('target ' + y[cur.i], outX - 4 * sc, midY + 43 * sc);
    }
  }

  /* ─── Drawing: plane ─── */
  var v = M.view(plane, { x: [-1.3, 1.3], y: [-1.3, 1.3], equal: true, pad: { l: 12, r: 12, t: 12, b: 12 } });
  function drawPlane() {
    var ctx = plane.ctx, s = plane;
    if (!s.w) return;
    if (real) { var bb = M.bounds(X, 0.15); v.setDomain(bb.x, bb.y); } else v.update();
    M.clear(ctx, s);
    M.field(ctx, v, 60, 60, function (x, yy) {
      var c = act([x, yy]) >= 0 ? M.hex(C.orange) : M.hex(C.blue);
      return [c[0], c[1], c[2], 34];
    }, false);
    ctx.save(); v.clip(ctx);
    if (Math.abs(w[1]) > 1e-9 || Math.abs(w[0]) > 1e-9) {
      var p1, p2;
      if (Math.abs(w[1]) > Math.abs(w[0])) { p1 = [v.x0, -(w[2] + w[0] * v.x0) / w[1]]; p2 = [v.x1, -(w[2] + w[0] * v.x1) / w[1]]; }
      else { p1 = [-(w[2] + w[1] * v.y0) / w[0], v.y0]; p2 = [-(w[2] + w[1] * v.y1) / w[0], v.y1]; }
      ctx.strokeStyle = flash > 0 ? M.rgba(C.red, 0.5 + 0.5 * flash) : C.ink;
      ctx.lineWidth = (2 + 2 * flash) * s.sc;
      ctx.beginPath(); ctx.moveTo(v.px(p1[0]), v.py(p1[1])); ctx.lineTo(v.px(p2[0]), v.py(p2[1])); ctx.stroke();
    }
    for (var i = 0; i < X.length; i++) {
      var wrong = pred(X[i]) !== y[i];
      M.dot(ctx, v.px(X[i][0]), v.py(X[i][1]), 4.2 * s.sc, y[i] ? C.orange : C.blue, wrong ? C.red : C.surface, wrong ? 2 : 1);
    }
    if (cur) M.dot(ctx, v.px(cur.p[0]), v.py(cur.p[1]), 9 * s.sc, null, C.ink, 1.8 * s.sc);
    ctx.restore();
  }

  function drawAll() {
    drawNet(); drawPlane();
    var wrongNow = 0;
    for (var i = 0; i < X.length; i++) if (pred(X[i]) !== y[i]) wrongNow++;
    var msg = {
      ready: '<span class="k">press play: the network will look at one point at a time</span>',
      training: cur && cur.err ? '<span class="o">mistake! the weights get nudged toward the right answer</span>' : '<span class="b">learning&hellip;</span>',
      converged: '<span class="g">a full pass with zero mistakes: the perceptron has converged</span>',
      stuck: '<span class="r">never converges: no single straight line can split ' + (presetT && presetT.get() === 'xor' ? 'XOR' : 'these points') +
        '. That takes hidden layers (Part 2!)</span>'
    }[status];
    read.innerHTML = '<span class="k">pass</span> ' + (epoch + 1) +
      '<span class="sep">&middot;</span><span class="k">points on the wrong side</span> ' + wrongNow + ' / ' + X.length +
      (lastMistakes != null ? '<span class="sep">&middot;</span><span class="k">mistakes last pass</span> ' + lastMistakes : '') +
      '<span class="sep">&middot;</span>' + msg;
  }

  M.pointer(plane.cv, {
    down: function (p, e) {
      if (!v.inside(p.x, p.y)) return false;
      X.push([v.ix(p.x), v.iy(p.y)]); y.push(e.shiftKey ? 0 : 1);
      if (status === 'converged' || status === 'stuck') { status = 'ready'; epoch = 0; lastMistakes = null; }
      newOrder();
      drawAll();
      return false;
    },
    hover: function (p) { plane.cv.style.cursor = v.inside(p.x, p.y) ? 'copy' : ''; }
  });

  load();
  net.draw = drawAll; plane.draw = drawAll;
  net.request();
  return { destroy: function () { pause(); stepLoop.stop(); } };
});
