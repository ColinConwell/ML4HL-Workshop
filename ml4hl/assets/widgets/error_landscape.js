/* ══════════════════════════════════════════════════════════════════
   Figure 1 — Learning is rolling downhill (error landscape + descent)
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('error_landscape', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Learning is rolling downhill',
    hint: 'Drag the ball &middot; press play &middot; change the step size'
  });

  var seed = cfg.seed || 3, wells = cfg.wells || 3;
  var START = cfg.start == null ? 0.03 : cfg.start;
  var N = 600, SNAP = 0.012, STEP_T = 0.17, MAX_STEPS = 240;
  var land, grid, minima, globalIdx;

  function build() {
    var rng = M.rng(seed * 97 + wells);
    var bowlC = rng.uniform(0.38, 0.62), bowlA = rng.uniform(0.6, 1.0);
    var ws = [];
    for (var i = 0; i < wells; i++) {
      ws.push({
        m: (i + 0.5) / wells + rng.uniform(-0.25, 0.25) / wells,
        d: rng.uniform(0.25, 0.75),
        w: rng.uniform(0.035, 0.075)
      });
    }
    ws[0].d = Math.min(ws[0].d, 0.6);
    ws[1 + rng.int(wells - 1)].d = 0.95;
    land = function (x) {
      var v = bowlA * (x - bowlC) * (x - bowlC);
      for (var k = 0; k < ws.length; k++) {
        var z = (x - ws[k].m) / ws[k].w;
        v -= ws[k].d * Math.exp(-0.5 * z * z) * 0.6;
      }
      return v;
    };
    grid = new Float64Array(N + 1);
    var lo = Infinity, hi = -Infinity;
    for (i = 0; i <= N; i++) { grid[i] = land(i / N); lo = Math.min(lo, grid[i]); hi = Math.max(hi, grid[i]); }
    var raw = land;
    land = function (x) { return (raw(x) - lo) / (hi - lo); };
    for (i = 0; i <= N; i++) grid[i] = (grid[i] - lo) / (hi - lo);
    minima = [];
    for (i = 1; i < N; i++) {
      if (grid[i] < grid[i - 1] && grid[i] <= grid[i + 1]) {
        var x = i / N;
        for (var it = 0; it < 30; it++) x = M.clamp(x - 0.002 * slope(x) / Math.max(1, Math.abs(curv(x))), 0, 1);
        minima.push({ x: x, y: land(x) });
      }
    }
    globalIdx = M.argmin(minima.map(function (m) { return m.y; }));
  }
  function slope(x) { var e = 1e-4; return (land(Math.min(1, x + e)) - land(Math.max(0, x - e))) / (2 * e); }
  function curv(x) { var e = 1e-3; return (land(x + e) - 2 * land(x) + land(x - e)) / (e * e); }

  /* ─── State ─── */
  var ball, from, to, tween, steps, history, trail, status, snapped, dragging = false;
  function resetBall(x) {
    ball = from = to = M.clamp(x, 0, 1);
    tween = 1; steps = 0; history = [land(ball)]; trail = []; status = 'ready'; snapped = false;
  }

  function nearestMin(x) {
    var best = 0;
    for (var i = 1; i < minima.length; i++) if (Math.abs(minima[i].x - x) < Math.abs(minima[best].x - x)) best = i;
    return best;
  }

  function stepOnce() {
    var lr = lrSlider.get();
    var g = slope(ball);
    var nx = ball - lr * g, bounced = false;
    if (nx < 0) { nx = -nx * 0.5; bounced = true; }
    if (nx > 1) { nx = 1 - (nx - 1) * 0.5; bounced = true; }
    nx = M.clamp(nx, 0, 1);
    trail.push(ball);
    if (trail.length > 40) trail.shift();
    from = ball; to = nx; tween = 0; ball = nx;
    steps++;
    history.push(land(ball));
    var moved = Math.abs(to - from);
    if (moved < 2e-4 && Math.abs(g) < 0.05) {
      status = nearestMin(ball) === globalIdx ? 'global' : 'local';
      return false;
    }
    if (steps >= MAX_STEPS) { status = 'bouncing'; return false; }
    status = (bounced || (steps > 25 && moved > 0.08)) ? 'wild' : 'rolling';
    return true;
  }

  /* ─── DOM ─── */
  var stage = M.stage(root, { aspect: 0.43, minH: 230, maxH: 380, label: 'Error landscape with a ball descending by gradient descent' });
  var row1 = M.row(root);
  var playBtn = M.button(row1, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Play or pause' });
  M.button(row1, 'step', function () { pause(); if (status === 'global' || status === 'local') return; tween = 1; stepOnce(); runTween(); }, { aria: 'Take one gradient step' });
  var lrSlider = M.slider(row1, {
    label: 'Step size', min: 0.001, max: 0.08, value: cfg.lr || 0.006, log: true, accent: 'green',
    fmt: function (v) { return v.toFixed(4); },
    onInput: function () { if (status === 'local' || status === 'global' || status === 'bouncing') status = 'ready'; draw(); }
  });
  var row2 = M.row(root);
  M.button(row2, 'drop ball', function () { pause(); resetBall(M.rng(Date.now() & 0xffff).uniform(0.03, 0.97)); draw(); }, { aria: 'Drop the ball somewhere random' });
  M.button(row2, 'new landscape', function () { pause(); seed++; build(); resetBall(START); draw(); });
  M.spacer(row2);
  M.resetButton(row2, function () { pause(); resetBall(START); draw(); });
  var read = M.readout(root);

  var tweenLoop = M.loop(function (dt) {
    tween = Math.min(1, tween + dt / (STEP_T * 0.85));
    draw();
    return tween < 1;
  });
  function runTween() { tweenLoop.start(); }

  var acc = 0;
  var loop = M.loop(function (dt) {
    acc += dt;
    if (tween >= 1 && acc >= STEP_T) {
      acc = 0;
      if (!stepOnce()) { pause(); draw(); return false; }
    }
    tween = Math.min(1, tween + dt / (STEP_T * 0.85));
    draw();
    return true;
  });
  function play() {
    if (status === 'global' || status === 'local' || status === 'bouncing') resetBall(ball);
    acc = STEP_T; loop.start();
    playBtn.textContent = 'pause'; playBtn.classList.add('on');
    if (status === 'ready') status = 'rolling';
  }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  /* ─── Drawing ─── */
  var v = M.view(stage, { x: [0, 1], y: [-0.3, 1.1], pad: { l: 26, r: 14, t: 14, b: 28 } });

  function shownX() { return tween >= 1 ? ball : M.lerp(from, to, M.ease(tween)); }

  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w) return;
    v.update();
    M.clear(ctx, s);
    M.axes(ctx, v, s, { xlabels: false, ylabels: false, xlabel: 'weight (a knob the model can turn)', ylabel: 'error' });

    /* Landscape: filled area plus outline. */
    ctx.save();
    v.clip(ctx);
    var grad = ctx.createLinearGradient(0, v.T, 0, v.B);
    grad.addColorStop(0, 'rgba(37,99,235,0.02)');
    grad.addColorStop(1, 'rgba(37,99,235,0.13)');
    ctx.beginPath();
    ctx.moveTo(v.px(0), v.B);
    for (var i = 0; i <= N; i++) ctx.lineTo(v.px(i / N), v.py(grid[i]));
    ctx.lineTo(v.px(1), v.B);
    ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath();
    for (i = 0; i <= N; i++) ctx[i ? 'lineTo' : 'moveTo'](v.px(i / N), v.py(grid[i]));
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2 * s.sc; ctx.lineJoin = 'round'; ctx.stroke();
    ctx.restore();

    /* Minima markers. */
    ctx.font = M.font(s, 9.5, 500);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    minima.forEach(function (m, k) {
      var px = v.px(m.x), py = v.py(m.y);
      if (k === globalIdx) {
        M.star(ctx, px, py + 13 * s.sc, 7 * s.sc, C.gold, '#fff');
        ctx.fillStyle = C.gold;
        ctx.fillText('global minimum', px, py + 23 * s.sc);
      } else {
        M.dot(ctx, px, py + 12 * s.sc, 3.2 * s.sc, null, C.muted, 1.2 * s.sc);
        ctx.fillStyle = C.muted;
        ctx.fillText('local minimum', px, py + 19 * s.sc);
      }
    });

    /* Trail of previous positions. */
    trail.forEach(function (x, k) {
      var a = 0.12 + 0.5 * (k / trail.length);
      M.dot(ctx, v.px(x), v.py(land(x)), 3 * s.sc, M.rgba(C.blue, a));
    });

    /* Slope (gradient) at the ball, drawn as a tangent with a downhill arrow. */
    var bx = shownX(), by = land(bx), g = slope(bx);
    var px0 = v.px(bx), py0 = v.py(by);
    var dxp = v.R - v.L, dyp = v.B - v.T;
    var ang = Math.atan2(-g * dyp / (v.y1 - v.y0), dxp / (v.x1 - v.x0));
    var L = 34 * s.sc;
    ctx.strokeStyle = M.rgba(C.green, 0.55); ctx.lineWidth = 1.3 * s.sc;
    ctx.setLineDash([4 * s.sc, 3 * s.sc]);
    ctx.beginPath();
    ctx.moveTo(px0 - Math.cos(ang) * L, py0 - Math.sin(ang) * L);
    ctx.lineTo(px0 + Math.cos(ang) * L, py0 + Math.sin(ang) * L);
    ctx.stroke(); ctx.setLineDash([]);
    if (Math.abs(g) > 0.03 && !dragging) {
      var dir = g > 0 ? -1 : 1, al = M.clamp(Math.abs(g) * 9, 12, 46) * s.sc;
      M.arrow(ctx, px0 + dir * Math.cos(ang) * 10 * s.sc, py0 + dir * Math.sin(ang) * 10 * s.sc,
        px0 + dir * Math.cos(ang) * (10 * s.sc + al), py0 + dir * Math.sin(ang) * (10 * s.sc + al), C.green, 2 * s.sc, 7 * s.sc);
    }

    /* The ball. */
    var col = status === 'local' ? C.orange : status === 'global' ? C.gold : status === 'wild' || status === 'bouncing' ? C.red : C.blue;
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.18)'; ctx.shadowBlur = 5 * s.sc; ctx.shadowOffsetY = 1.5 * s.sc;
    M.dot(ctx, px0, py0 - 7.5 * s.sc, 7.5 * s.sc, col, '#fff', 2 * s.sc);
    ctx.restore();

    drawSpark(ctx, s);
    updateReadout(bx, by, g);
  }

  function drawSpark(ctx, s) {
    if (history.length < 2) return;
    var w = 132 * s.sc, ht = 46 * s.sc, x0 = v.R - w - 6 * s.sc, y0 = v.T + 6 * s.sc;
    ctx.save();
    ctx.fillStyle = M.rgba(C.surface, 0.92); ctx.strokeStyle = C.border; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.rect(x0, y0, w, ht); ctx.fill(); ctx.stroke();
    ctx.font = M.font(s, 8.5, 500); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('ERROR OVER STEPS', x0 + 6 * s.sc, y0 + 5 * s.sc);
    var n = history.length, hi = Math.max.apply(null, history), lo = Math.min.apply(null, history.concat([0]));
    var ix = function (i) { return x0 + 6 * s.sc + (i / Math.max(1, n - 1)) * (w - 12 * s.sc); };
    var iy = function (e) { return y0 + ht - 6 * s.sc - (e - lo) / ((hi - lo) || 1) * (ht - 22 * s.sc); };
    ctx.beginPath();
    for (var i = 0; i < n; i++) ctx[i ? 'lineTo' : 'moveTo'](ix(i), iy(history[i]));
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.4 * s.sc; ctx.stroke();
    M.dot(ctx, ix(n - 1), iy(history[n - 1]), 2.4 * s.sc, C.blue);
    ctx.restore();
  }

  function updateReadout(x, e, g) {
    var msg = {
      ready: '<span class="k">ready: press play</span>',
      rolling: '<span class="b">rolling downhill&hellip;</span>',
      wild: '<span class="r">overshooting: each step jumps past the valley floor</span>',
      bouncing: '<span class="r">bouncing forever: the step is too big to ever settle</span>',
      local: '<span class="o">stuck in a local minimum</span> <span class="k">&mdash; try a bigger step, or drop the ball elsewhere</span>',
      global: '<span class="g">settled in the global minimum</span>'
    }[status];
    read.innerHTML = '<span class="k">step</span> ' + steps +
      '<span class="sep">&middot;</span><span class="k">error</span> ' + e.toFixed(3) +
      '<span class="sep">&middot;</span><span class="k">slope</span> ' + M.f(g, 2) +
      (snapped ? ' <span class="k">(snapped)</span>' : '') +
      '<span class="sep">&middot;</span>' + msg;
  }

  /* ─── Interaction ─── */
  function hitBall(p) {
    var bx = shownX();
    return Math.hypot(p.x - v.px(bx), p.y - (v.py(land(bx)) - 7.5 * stage.sc)) < 16 * stage.sc;
  }
  function placeAt(px) {
    var x = M.clamp(v.ix(px), 0, 1);
    snapped = false;
    minima.forEach(function (m) { if (Math.abs(m.x - x) < SNAP) { x = m.x; snapped = true; } });
    resetBall(x);
    snapped = minima.some(function (m) { return m.x === x; });
  }
  M.pointer(stage.cv, {
    down: function (p) {
      pause();
      if (!hitBall(p)) { if (!v.inside(p.x, p.y)) return false; }
      dragging = true; placeAt(p.x); stage.cv.style.cursor = 'grabbing'; draw();
    },
    drag: function (p) { placeAt(p.x); draw(); },
    up: function () { dragging = false; stage.cv.style.cursor = 'grab'; draw(); },
    hover: function (p) { stage.cv.style.cursor = hitBall(p) ? 'grab' : 'pointer'; },
    key: function (e) {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        pause(); resetBall(M.clamp(ball + (e.key === 'ArrowLeft' ? -0.01 : 0.01), 0, 1)); draw(); return true;
      }
      if (e.key === ' ') { loop.on ? pause() : play(); return true; }
    }
  });

  build();
  resetBall(START);
  stage.draw = draw;
  stage.request();
  return {
    destroy: function () { pause(); tweenLoop.stop(); },
    state: function () { return { ball: ball, steps: steps, status: status }; }
  };
});
