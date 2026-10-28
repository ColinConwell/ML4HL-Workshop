/* ══════════════════════════════════════════════════════════════════
   Figure 2 — Fit a line by eye, then by least squares
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('ols_fit', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Fit a line by eye, then by least squares',
    hint: 'Drag the two handles &middot; or drag the line itself'
  });

  var real = cfg.data && cfg.data.x;
  var seed = cfg.seed || 7;
  var xs, ys, dom, ols, h1, h2, hx1, hx2, ghost = false, anim = null;

  function load() {
    if (real) { xs = cfg.data.x.slice(); ys = cfg.data.y.slice(); }
    else {
      var d = M.sim.linear(M.rng(seed), { n: cfg.n || 28, noise: cfg.noise == null ? 0.6 : cfg.noise, slope: cfg.slope == null ? 0.8 : cfg.slope, intercept: 0.4 });
      xs = d.x; ys = d.y;
    }
    var X = xs.map(function (x, i) { return [x, ys[i]]; });
    dom = M.bounds(X, 0.12);
    ols = M.linfit(xs, ys);
    var span = dom.x[1] - dom.x[0];
    hx1 = dom.x[0] + span * 0.14; hx2 = dom.x[0] + span * 0.86;
    flat();
  }
  function flat() { var my = M.mean(ys); h1 = my; h2 = my; }

  function lineAt(x) { return h1 + (h2 - h1) * (x - hx1) / (hx2 - hx1); }
  function sse(fn) { var s = 0; for (var i = 0; i < xs.length; i++) { var r = ys[i] - fn(xs[i]); s += r * r; } return s; }
  function olsAt(x) { return ols.m * x + ols.b; }

  /* ─── DOM ─── */
  var stage = M.stage(root, { aspect: 0.5, minH: 250, maxH: 400, label: 'Scatter plot with a draggable line and its squared residuals' });
  var row = M.row(root);
  var resToggle = M.toggle(row, {
    label: 'Residuals', value: cfg.residuals || (real && cfg.data.x.length > 50 ? 'lines' : 'squares'),
    options: [['squares', 'squares'], ['lines', 'lines']],
    onChange: function () { draw(); }
  });
  var ghostBtn = M.button(row, 'show best line', function () { ghost = !ghost; ghostBtn.classList.toggle('on', ghost); draw(); }, { aria: 'Toggle the least-squares line' });
  M.button(row, 'snap to least squares', function () { animateTo(olsAt(hx1), olsAt(hx2)); }, { cls: 'wplay', aria: 'Animate the line to the least-squares solution' });
  M.spacer(row);
  if (!real) M.button(row, 'new data', function () { seed++; load(); draw(); }, { aria: 'Draw a new random dataset' });
  M.resetButton(row, function () { stopAnim(); flat(); draw(); });
  var read = M.readout(root);

  function stopAnim() { if (anim) anim.stop(); anim = null; }
  function animateTo(t1, t2) {
    stopAnim();
    var s1 = h1, s2 = h2, t = 0;
    anim = M.loop(function (dt) {
      t = Math.min(1, t + dt / 0.7);
      var e = M.ease(t);
      h1 = M.lerp(s1, t1, e); h2 = M.lerp(s2, t2, e);
      draw();
      return t < 1;
    });
    anim.start();
  }

  /* ─── Drawing ─── */
  var v = M.view(stage, { x: [0, 1], y: [0, 1], pad: real ? { l: 56, b: 40 } : { l: 40, b: 30 } });

  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w || !xs) return;
    v.setDomain(dom.x, dom.y);
    M.clear(ctx, s);
    M.axes(ctx, v, s, {
      xlabel: real ? (cfg.data.xlabel || 'x') : 'x', ylabel: real ? (cfg.data.ylabel || 'y') : 'y',
      xlabels: !!real, ylabels: !!real
    });

    ctx.save();
    v.clip(ctx);
    var squares = resToggle.get() === 'squares';
    var midX = (dom.x[0] + dom.x[1]) / 2;
    for (var i = 0; i < xs.length; i++) {
      var px = v.px(xs[i]), py = v.py(ys[i]), pl = v.py(lineAt(xs[i]));
      var side = Math.abs(py - pl);
      if (squares) {
        var dir = xs[i] > midX ? -1 : 1;
        ctx.fillStyle = M.rgba(C.orange, 0.11);
        ctx.strokeStyle = M.rgba(C.orange, 0.5);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(dir > 0 ? px : px - side, Math.min(py, pl), side, side);
        ctx.fill(); ctx.stroke();
      } else {
        ctx.strokeStyle = M.rgba(C.orange, 0.75); ctx.lineWidth = 1.4 * s.sc;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, pl); ctx.stroke();
      }
    }
    if (ghost) {
      ctx.setLineDash([6 * s.sc, 4 * s.sc]);
      ctx.strokeStyle = C.green; ctx.lineWidth = 1.6 * s.sc;
      ctx.beginPath(); ctx.moveTo(v.px(dom.x[0]), v.py(olsAt(dom.x[0]))); ctx.lineTo(v.px(dom.x[1]), v.py(olsAt(dom.x[1]))); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.4 * s.sc;
    ctx.beginPath(); ctx.moveTo(v.px(dom.x[0]), v.py(lineAt(dom.x[0]))); ctx.lineTo(v.px(dom.x[1]), v.py(lineAt(dom.x[1]))); ctx.stroke();
    for (i = 0; i < xs.length; i++) M.dot(ctx, v.px(xs[i]), v.py(ys[i]), 3.6 * s.sc, C.ink, '#fff', 1);
    [[hx1, h1], [hx2, h2]].forEach(function (hd) {
      M.dot(ctx, v.px(hd[0]), v.py(hd[1]), 8 * s.sc, '#fff', C.blue, 2.4 * s.sc);
      M.dot(ctx, v.px(hd[0]), v.py(hd[1]), 3 * s.sc, C.blue);
    });
    ctx.restore();

    drawMeter(ctx, s);
    updateReadout();
  }

  function drawMeter(ctx, s) {
    var yours = sse(lineAt), best = sse(olsAt), worst = Math.max(yours, sse(function () { return M.mean(ys); })) * 1.05;
    var w = 150 * s.sc, y0 = v.T + 8 * s.sc, bh = 7 * s.sc;
    var boxW = w + 16 * s.sc, boxH = 48 * s.sc, inL = 0, inR = 0;
    for (var i = 0; i < xs.length; i++) {
      var px = v.px(xs[i]), py = v.py(ys[i]);
      if (py < v.T + boxH) { if (px < v.L + boxW + 8 * s.sc) inL++; if (px > v.R - boxW - 8 * s.sc) inR++; }
    }
    var x0 = inL < inR ? v.L + 16 * s.sc : v.R - w - 8 * s.sc;
    ctx.save();
    ctx.fillStyle = M.rgba(C.surface, 0.94); ctx.strokeStyle = C.border;
    ctx.beginPath(); ctx.rect(x0 - 8 * s.sc, y0 - 4 * s.sc, w + 16 * s.sc, 40 * s.sc); ctx.fill(); ctx.stroke();
    ctx.font = M.font(s, 8.5, 500); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('TOTAL SQUARE AREA', x0, y0);
    var by = y0 + 15 * s.sc;
    ctx.fillStyle = C.grid; ctx.fillRect(x0, by, w, bh);
    ctx.fillStyle = M.rgba(C.orange, 0.85); ctx.fillRect(x0, by, w * Math.min(1, yours / worst), bh);
    var bx = x0 + w * best / worst;
    ctx.strokeStyle = C.green; ctx.lineWidth = 2 * s.sc;
    ctx.beginPath(); ctx.moveTo(bx, by - 3 * s.sc); ctx.lineTo(bx, by + bh + 3 * s.sc); ctx.stroke();
    ctx.font = M.font(s, 8, 500); ctx.fillStyle = C.green; ctx.textAlign = 'center';
    ctx.fillText('best', bx, by + bh + 4 * s.sc);
    ctx.restore();
  }

  function updateReadout() {
    var yours = sse(lineAt), best = sse(olsAt), over = (yours - best) / best;
    var verdict = over < 0.01
      ? '<span class="g">that is the least-squares line: no other line has less square area</span>'
      : over < 0.15 ? '<span class="b">close! ' + Math.round(over * 100) + '% more area than the best line</span>'
      : '<span class="o">' + Math.round(over * 100) + '% more square area than the best possible line</span>';
    var slope = (h2 - h1) / (hx2 - hx1);
    read.innerHTML = '<span class="k">your slope</span> ' + M.f(slope, 2) +
      '<span class="sep">&middot;</span><span class="k">your area</span> ' + M.f(yours, yours > 100 ? 0 : 2) +
      '<span class="sep">&middot;</span><span class="k">best</span> ' + M.f(best, best > 100 ? 0 : 2) +
      '<span class="sep">&middot;</span>' + verdict;
  }

  /* ─── Interaction ─── */
  var dragging = null, grabY = 0, grab1 = 0, grab2 = 0;
  function hit(p) {
    if (Math.hypot(p.x - v.px(hx1), p.y - v.py(h1)) < 14 * stage.sc) return 'h1';
    if (Math.hypot(p.x - v.px(hx2), p.y - v.py(h2)) < 14 * stage.sc) return 'h2';
    var x = v.ix(p.x);
    if (Math.abs(p.y - v.py(lineAt(x))) < 9 * stage.sc) return 'line';
    return null;
  }
  M.pointer(stage.cv, {
    down: function (p) {
      var hh = hit(p);
      if (!hh) return false;
      stopAnim();
      dragging = hh; grabY = v.iy(p.y); grab1 = h1; grab2 = h2;
      stage.cv.style.cursor = 'grabbing';
    },
    drag: function (p) {
      var y = M.clamp(v.iy(p.y), dom.y[0], dom.y[1]);
      if (dragging === 'h1') h1 = y;
      else if (dragging === 'h2') h2 = y;
      else { h1 = grab1 + (y - grabY); h2 = grab2 + (y - grabY); }
      var o1 = olsAt(hx1), o2 = olsAt(hx2), tol = (dom.y[1] - dom.y[0]) * 0.012;
      if (Math.abs(h1 - o1) < tol && Math.abs(h2 - o2) < tol) { h1 = o1; h2 = o2; }
      draw();
    },
    up: function () { dragging = null; stage.cv.style.cursor = ''; },
    hover: function (p) { var hh = hit(p); stage.cv.style.cursor = hh === 'line' ? 'ns-resize' : hh ? 'grab' : ''; },
    key: function (e) {
      var step = (dom.y[1] - dom.y[0]) * 0.02;
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        var d = e.key === 'ArrowUp' ? step : -step;
        if (e.shiftKey) h2 += d; else h1 += d;
        draw(); return true;
      }
    }
  });

  load();
  stage.draw = draw;
  stage.request();
  return { destroy: stopAnim };
});
