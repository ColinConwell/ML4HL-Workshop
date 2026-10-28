/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 6 — Folding the plane: a 2-layer ReLU network
   solves XOR
   A single perceptron can only draw one straight line and fails on XOR.
   Adding a hidden layer of ReLU hinges folds the input plane so the
   opposite corners land on the same side of a linear readout.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('mlp_xor', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Folding the plane: how one hidden layer solves XOR',
    hint: 'Press play to train &middot; watch the hidden hinges fold the four corners into a separable cloud'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 4;
  var X, y, H, W1, b1, W2, b2, stepCount = 0, accNow = 0;

  function load() {
    var rng = M.rng(seed);
    if (real) {
      X = cfg.data.X.map(function (p) { return p.slice(); });
      y = cfg.data.y.slice();
    } else {
      X = []; y = [];
      var preset = presetT ? presetT.get() : (cfg.preset || 'xor');
      if (preset === 'circles') {
        var d = M.sim.circles(rng, { n: 36, noise: 0.08 });
        X = d.X; y = d.y;
      } else {
        [[-0.65, -0.65, 0], [0.65, 0.65, 0], [-0.65, 0.65, 1], [0.65, -0.65, 1]].forEach(function (c) {
          for (var i = 0; i < 14; i++) {
            X.push([c[0] + rng.normal(0, 0.16), c[1] + rng.normal(0, 0.16)]);
            y.push(c[2]);
          }
        });
      }
    }
    initNet();
  }

  function initNet() {
    H = hS ? hS.get() : (cfg.hidden || 4);
    var rng = M.rng(seed * 19 + H);
    W1 = []; b1 = new Float64Array(H); W2 = new Float64Array(H); b2 = 0;
    for (var j = 0; j < H; j++) {
      var a = (2 * Math.PI * j) / H + rng.uniform(-0.2, 0.2);
      W1.push([1.4 * Math.cos(a), 1.4 * Math.sin(a)]);
      b1[j] = rng.uniform(-0.3, 0.3);
      W2[j] = (j % 2 ? 1 : -1) * rng.uniform(0.5, 1.1);
    }
    stepCount = 0;
    evalAcc();
  }

  function hiddenAct(p) {
    var h = new Float64Array(H);
    for (var j = 0; j < H; j++) {
      var z = W1[j][0] * p[0] + W1[j][1] * p[1] + b1[j];
      h[j] = z > 0 ? z : 0.05 * z;
    }
    return h;
  }

  function forward(p) {
    var h = hiddenAct(p), s = b2;
    for (var j = 0; j < H; j++) s += W2[j] * h[j];
    return { h: h, prob: M.sigmoid(s) };
  }

  function trainStep(iters) {
    var lr = 0.18, n = X.length;
    for (var it = 0; it < iters; it++) {
      var gW1 = W1.map(function () { return [0, 0]; }), gb1 = new Float64Array(H);
      var gW2 = new Float64Array(H), gb2 = 0;
      for (var i = 0; i < n; i++) {
        var fw = forward(X[i]), err = fw.prob - y[i];
        gb2 += err;
        for (var j = 0; j < H; j++) {
          gW2[j] += err * fw.h[j];
          var z = W1[j][0] * X[i][0] + W1[j][1] * X[i][1] + b1[j];
          var dh = err * W2[j] * (z > 0 ? 1 : 0.05);
          gW1[j][0] += dh * X[i][0];
          gW1[j][1] += dh * X[i][1];
          gb1[j] += dh;
        }
      }
      b2 -= lr * (gb2 / n);
      for (var j2 = 0; j2 < H; j2++) {
        W2[j2] -= lr * (gW2[j2] / n + 0.001 * W2[j2]);
        W1[j2][0] -= lr * (gW1[j2][0] / n + 0.001 * W1[j2][0]);
        W1[j2][1] -= lr * (gW1[j2][1] / n + 0.001 * W1[j2][1]);
        b1[j2] -= lr * (gb1[j2] / n);
      }
      stepCount++;
    }
    evalAcc();
  }

  function evalAcc() {
    var c = 0;
    for (var i = 0; i < X.length; i++) if ((forward(X[i]).prob >= 0.5 ? 1 : 0) === y[i]) c++;
    accNow = c / (X.length || 1);
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, 'input space: hidden hinge lines + combined boundary');
  M.mini(split.b, 'hidden representation (first 2 hidden units)');
  var hf = function (w) { return M.clamp(w * 0.84, 220, 330); };
  var left = M.stage(split.a, { height: hf, ref: 400, label: 'Input plane showing hidden unit hinge lines and nonlinear decision region' });
  var right = M.stage(split.b, { height: hf, ref: 400, label: 'Hidden-layer representation where XOR becomes linearly separable' });

  var legend = M.legend(root, [
    { color: C.blue, label: 'class A (0)' },
    { color: C.orange, label: 'class B (1)' },
    { color: C.muted, label: 'hidden ReLU hinge lines', line: true },
    { color: C.ink, label: 'combined boundary (50%)', line: true }
  ]);
  var row = M.row(root);
  var playBtn = M.button(row, 'play', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Train or pause the 2-layer network' });
  M.button(row, 'step', function () { pause(); trainStep(8); drawAll(); }, { aria: 'Run 8 gradient steps' });
  var hS = M.slider(row, {
    label: 'Hidden units', min: 2, max: 8, step: 1, value: cfg.hidden || 4, accent: 'blue',
    onInput: function () { pause(); initNet(); drawAll(); }
  });
  var presetT = null;
  if (!real) {
    presetT = M.toggle(row, {
      label: 'Pattern', value: cfg.preset || 'xor',
      options: [['xor', 'XOR'], ['circles', 'rings']],
      onChange: function () { pause(); load(); drawAll(); }
    });
  }
  M.spacer(row);
  M.resetButton(row, function () { pause(); hS.set(cfg.hidden || 4, true); if (presetT) presetT.set(cfg.preset || 'xor', true); load(); drawAll(); });
  var read = M.readout(root);

  var loop = M.loop(function () {
    trainStep(6);
    drawAll();
    if (stepCount >= 600 || accNow >= 0.995) { pause(); return false; }
    return true;
  });
  function play() { if (stepCount >= 600 || accNow >= 0.995) initNet(); loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'play'; playBtn.classList.remove('on'); }

  /* ─── Drawing ─── */
  var vL = M.view(left, { x: [-1.35, 1.35], y: [-1.35, 1.35], equal: true, pad: { l: 14, r: 14, t: 14, b: 14 } });
  var vR = M.view(right, { x: [0, 1], y: [0, 1], pad: { l: 38, r: 14, t: 14, b: 34 } });

  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    if (real) { var bb = M.bounds(X, 0.15); vL.setDomain(bb.x, bb.y); } else vL.setDomain([-1.35, 1.35], [-1.35, 1.35]);
    M.clear(ctx, s);
    M.field(ctx, vL, 64, 64, function (x, yy) {
      var p = forward([x, yy]).prob, c = M.diverge(p);
      return [c[0], c[1], c[2], 145];
    });

    ctx.save(); vL.clip(ctx);
    /* Hidden hinge lines: W1[j][0]*x + W1[j][1]*y + b1[j] = 0 */
    ctx.setLineDash([4 * s.sc, 3 * s.sc]); ctx.lineWidth = 1.2 * s.sc;
    for (var j = 0; j < H; j++) {
      var w = W1[j], b = b1[j];
      ctx.strokeStyle = M.rgba(C.ink, 0.42);
      ctx.beginPath();
      if (Math.abs(w[1]) > Math.abs(w[0])) {
        ctx.moveTo(vL.px(vL.x0), vL.py(-(b + w[0] * vL.x0) / w[1]));
        ctx.lineTo(vL.px(vL.x1), vL.py(-(b + w[0] * vL.x1) / w[1]));
      } else if (Math.abs(w[0]) > 1e-6) {
        ctx.moveTo(vL.px(-(b + w[1] * vL.y0) / w[0]), vL.py(vL.y0));
        ctx.lineTo(vL.px(-(b + w[1] * vL.y1) / w[0]), vL.py(vL.y1));
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    for (var i = 0; i < X.length; i++) {
      var wrong = (forward(X[i]).prob >= 0.5 ? 1 : 0) !== y[i];
      M.dot(ctx, vL.px(X[i][0]), vL.py(X[i][1]), 4 * s.sc, y[i] ? C.orange : C.blue, wrong ? C.red : '#fff', wrong ? 2 : 1);
    }
    ctx.restore();
  }

  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    var H2 = X.map(function (p) { var h = hiddenAct(p); return [h[0], h[1]]; });
    var bb = M.bounds(H2, 0.16);
    vR.setDomain(bb.x, bb.y);
    M.clear(ctx, s);
    M.axes(ctx, vR, s, { xlabel: 'hidden unit h\u2081', ylabel: 'hidden unit h\u2082', nx: 3, ny: 3 });

    ctx.save(); vR.clip(ctx);
    for (var i = 0; i < H2.length; i++) {
      M.dot(ctx, vR.px(H2[i][0]), vR.py(H2[i][1]), 3.8 * s.sc, y[i] ? C.orange : C.blue, '#fff', 0.9);
    }
    ctx.restore();
  }

  function drawAll() {
    drawLeft(); drawRight();
    var msg = accNow >= 0.96
      ? '<span class="g">solved: ' + H + ' hidden ReLU hinges fold the plane so a linear readout separates the classes</span>'
      : stepCount === 0
        ? '<span class="k">press play: each dashed line is one hidden ReLU unit</span>'
        : '<span class="b">bending the boundary by moving the hidden hinges&hellip;</span>';
    read.innerHTML = '<span class="k">step</span> ' + stepCount +
      '<span class="sep">&middot;</span><span class="k">hidden hinges</span> ' + H +
      '<span class="sep">&middot;</span><span class="k">accuracy</span> ' + Math.round(accNow * 100) + '%' +
      '<span class="sep">&middot;</span>' + msg;
  }

  load();
  left.draw = drawAll; right.draw = drawAll;
  left.request();
  return { destroy: pause };
});
