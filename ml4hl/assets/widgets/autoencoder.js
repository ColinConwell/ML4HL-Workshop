/* ══════════════════════════════════════════════════════════════════
   Figure 11 — An autoencoder squeezes, then rebuilds
   A tiny network (64 -> 32 -> k -> 32 -> 64) trained live in the browser
   to copy 8x8 images through a k-number bottleneck. Drag the code bars to
   feed the decoder your own numbers.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('autoencoder', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'An autoencoder squeezes, then rebuilds',
    hint: 'Press train &middot; change the bottleneck &middot; drag a code bar to imagine'
  });

  var real = cfg.data && cfg.data.images;
  var side = (real && cfg.data.side) || 8, D = side * side, H = 32;
  var seed = cfg.seed || 3, X, colorOf, labels = null;

  /* ─── Data ─── */
  function load() {
    if (real) {
      X = cfg.data.images.map(function (r) { return Float64Array.from(r); });
      labels = cfg.data.labels || null;
      colorOf = function (i) { return labels ? M.catColor(labels[i]) : C.blue; };
    } else {
      // Soft blobs: three hidden factors (left-right, up-down, size) make every image.
      var rng = M.rng(seed), facts = [];
      X = [];
      for (var n = 0; n < 260; n++) {
        var cx = rng.uniform(1.2, side - 2.2), cy = rng.uniform(1.2, side - 2.2), r = rng.uniform(0.7, 1.9), img = new Float64Array(D);
        for (var y = 0; y < side; y++) for (var x = 0; x < side; x++) img[y * side + x] = Math.exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * r * r));
        X.push(img); facts.push(cx / side);
      }
      colorOf = function (i) { return M.css(M.magma(0.15 + 0.75 * facts[i])); };
    }
  }

  /* ─── Network ─── */
  var P, K, step, losses, epoch, adam = { b1: 0.9, b2: 0.999 };
  function param(rows, cols, rng) {
    var w = new Float64Array(rows * cols), s = Math.sqrt(2 / (rows + cols));
    if (cols > 1) for (var i = 0; i < w.length; i++) w[i] = rng.normal(0, s);
    return { w: w, g: new Float64Array(w.length), m: new Float64Array(w.length), v: new Float64Array(w.length), r: rows, c: cols };
  }
  function init() {
    K = kS.get();
    var rng = M.rng(seed * 31 + K);
    P = { W1: param(H, D, rng), b1: param(H, 1, rng), W2: param(K, H, rng), b2: param(K, 1, rng),
          W3: param(H, K, rng), b3: param(H, 1, rng), W4: param(D, H, rng), b4: param(D, 1, rng) };
    step = 0; losses = []; epoch = 0;
    code = new Float64Array(K);
    encodeAll();
    codeFromSample();
  }
  function matvec(p, x, b, out, act) {
    for (var i = 0; i < p.r; i++) {
      var s = b.w[i], o = i * p.c;
      for (var j = 0; j < p.c; j++) s += p.w[o + j] * x[j];
      out[i] = act === 'tanh' ? Math.tanh(s) : act === 'sig' ? 1 / (1 + Math.exp(-s)) : s;
    }
    return out;
  }
  function encode(x) { var h1 = matvec(P.W1, x, P.b1, new Float64Array(H), 'tanh'); return matvec(P.W2, h1, P.b2, new Float64Array(K)); }
  function decode(z) { var h2 = matvec(P.W3, z, P.b3, new Float64Array(H), 'tanh'); return matvec(P.W4, h2, P.b4, new Float64Array(D), 'sig'); }

  var h1 = new Float64Array(H), h2 = new Float64Array(H), out = new Float64Array(D);
  var d4 = new Float64Array(D), dh2 = new Float64Array(H), dz, dh1 = new Float64Array(H);
  function trainBatch(idx) {
    var z = new Float64Array(K); dz = new Float64Array(K);
    Object.keys(P).forEach(function (k) { P[k].g.fill(0); });
    var tot = 0;
    idx.forEach(function (n) {
      var x = X[n];
      matvec(P.W1, x, P.b1, h1, 'tanh'); matvec(P.W2, h1, P.b2, z); matvec(P.W3, z, P.b3, h2, 'tanh'); matvec(P.W4, h2, P.b4, out, 'sig');
      var i, j, o;
      for (i = 0; i < D; i++) { d4[i] = (out[i] - x[i]) / D; tot += (out[i] - x[i]) * (out[i] - x[i]) / D; }
      dh2.fill(0);
      for (i = 0; i < D; i++) { o = i * H; P.b4.g[i] += d4[i]; for (j = 0; j < H; j++) { P.W4.g[o + j] += d4[i] * h2[j]; dh2[j] += P.W4.w[o + j] * d4[i]; } }
      dz.fill(0);
      for (i = 0; i < H; i++) { var d3 = dh2[i] * (1 - h2[i] * h2[i]); o = i * K; P.b3.g[i] += d3; for (j = 0; j < K; j++) { P.W3.g[o + j] += d3 * z[j]; dz[j] += P.W3.w[o + j] * d3; } }
      dh1.fill(0);
      for (i = 0; i < K; i++) { o = i * H; P.b2.g[i] += dz[i]; for (j = 0; j < H; j++) { P.W2.g[o + j] += dz[i] * h1[j]; dh1[j] += P.W2.w[o + j] * dz[i]; } }
      for (i = 0; i < H; i++) { var d1 = dh1[i] * (1 - h1[i] * h1[i]); o = i * D; P.b1.g[i] += d1; for (j = 0; j < D; j++) P.W1.g[o + j] += d1 * x[j]; }
    });
    step++;
    var lr = lrS.get(), bc1 = 1 - Math.pow(adam.b1, step), bc2 = 1 - Math.pow(adam.b2, step);
    Object.keys(P).forEach(function (k) {
      var p = P[k];
      for (var i = 0; i < p.w.length; i++) {
        var g = p.g[i] / idx.length;
        p.m[i] = adam.b1 * p.m[i] + (1 - adam.b1) * g;
        p.v[i] = adam.b2 * p.v[i] + (1 - adam.b2) * g * g;
        p.w[i] -= lr * (p.m[i] / bc1) / (Math.sqrt(p.v[i] / bc2) + 1e-8);
      }
    });
    return tot / idx.length;
  }

  var order = [], cursor = 0, rngB = M.rng(99);
  function nextBatch(bs) {
    var b = [];
    for (var i = 0; i < bs; i++) {
      if (cursor >= order.length) { order = rngB.shuffle(M.range(X.length)); cursor = 0; epoch++; }
      b.push(order[cursor++]);
    }
    return b;
  }

  var codes = [];
  function encodeAll() { codes = X.slice(0, 260).map(encode); }

  /* ─── DOM ─── */
  var architecture = document.createElement('div');
  architecture.className = 'wae-architecture';
  root.appendChild(architecture);
  var architectureK = null, architectureNodes = [];
  function drawArchitecture() {
    if (!P) return;
    var widths = [D, H, K, H, D], names = ['Input', 'Encoder', 'Bottleneck', 'Decoder', 'Rebuilt'];
    if (architectureK !== K) {
      architectureK = K;
      var positions = widths.map(function (n, layer) {
        var count = Math.min(n, 6);
        return M.range(count).map(function (i) { return [52 + layer * 124, 76 + (i - (count - 1) / 2) * 18]; });
      });
      var svg = '<svg viewBox="0 0 600 180" role="img" aria-label="Autoencoder architecture: ' + D + ' input pixels, ' + H + ' encoder units, ' + K + (K === 1 ? ' bottleneck number, ' : ' bottleneck numbers, ') + H + ' decoder units, ' + D + ' reconstructed pixels. Representative units are shown; colors indicate current activations.">';
      positions.forEach(function (pts, layer) {
        if (layer < 4) pts.forEach(function (a) { positions[layer + 1].forEach(function (b) {
          svg += '<line x1="'+a[0]+'" y1="'+a[1]+'" x2="'+b[0]+'" y2="'+b[1]+'" class="wae-edge"/>';
        }); });
      });
      positions.forEach(function (pts, layer) {
        pts.forEach(function (point, i) { svg += '<circle cx="'+point[0]+'" cy="'+point[1]+'" r="6" data-layer="'+layer+'" data-unit="'+i+'"/>'; });
        svg += '<text x="'+(52+layer*124)+'" y="148">'+names[layer]+'</text><text x="'+(52+layer*124)+'" y="165">'+widths[layer]+(layer===0 || layer===4 ? ' pixels' : widths[layer] === 1 ? ' number' : ' numbers')+'</text>';
      });
      architecture.innerHTML = svg + '</svg><p>Input → compress → small code → rebuild. Representative units are shown; their colors follow the selected example’s actual activations.</p>';
      architectureNodes = Array.from(architecture.querySelectorAll('circle'));
    }
    var enc = matvec(P.W1, X[sample], P.b1, new Float64Array(H), 'tanh');
    var dec = matvec(P.W3, code, P.b3, new Float64Array(H), 'tanh');
    var values = [X[sample], enc, code, dec, decode(code)];
    architectureNodes.forEach(function (node) {
      var layer = +node.dataset.layer, unit = +node.dataset.unit;
      var value = values[layer][unit];
      node.style.fill = M.rgba(value < 0 ? C.blue : C.orange, 0.2 + 0.8 * Math.min(1, Math.abs(value)));
      node.style.stroke = C.ink;
    });
  }
  var split = M.split(root, 1.25);
  M.mini(split.a, 'originals (top) and rebuilt copies (bottom)');
  M.mini(split.b, 'the code space (first two code numbers)');
  var hf = function (w) { return M.clamp(w * 0.72, 230, 330); };
  var left = M.stage(split.a, { height: hf, ref: 440, label: 'Original and reconstructed images, plus code bars driving an imagined image' });
  var right = M.stage(split.b, { height: hf, ref: 360, label: 'Scatter plot of every image placed by its first two code numbers' });
  var row = M.row(root);
  var playBtn = M.button(row, 'train', function () { loop.on ? pause() : play(); }, { cls: 'wplay', role: 'play', aria: 'Train or pause the autoencoder' });
  var kS = M.slider(row, { label: 'Bottleneck (numbers)', min: 1, max: 16, step: 1, value: cfg.bottleneck || 3, accent: 'orange', onChange: function () { var was = loop.on; pause(); init(); draw(); if (was) play(); } });
  var row2 = M.row(root);
  var lrS = M.slider(row2, { label: 'Learning rate', min: 0.001, max: 0.05, step: 0.001, value: cfg.lr || 0.012, accent: 'blue', fmt: function (v) { return v.toFixed(3); } });
  M.spacer(row2);
  M.button(row2, 'next example', function () { sample = (sample + 1) % X.length; codeFromSample(); draw(); }, { aria: 'Show a different example in the imagine panel' });
  M.resetButton(row2, function () { pause(); kS.set(cfg.bottleneck || 3, true); lrS.set(cfg.lr || 0.012, true); sample = 0; load(); init(); draw(); });
  var read = M.readout(root);

  var sample = 0, code, dragBar = -1;
  function codeFromSample() { code = encode(X[sample]); }

  var loop = M.loop(function () {
    var l = 0;
    for (var i = 0; i < 6; i++) l += trainBatch(nextBatch(24));
    losses.push(l / 6);
    if (losses.length > 600) losses.splice(0, losses.length - 600);
    if (step % 30 === 0) encodeAll();
    if (dragBar < 0) codeFromSample();
    draw();
    return true;
  });
  function play() { loop.start(); playBtn.textContent = 'pause'; playBtn.classList.add('on'); }
  function pause() { loop.stop(); playBtn.textContent = 'train'; playBtn.classList.remove('on'); encodeAll(); }

  /* ─── Drawing ─── */
  function drawImg(ctx, img, x, y, sz, frame) {
    var px = sz / side;
    for (var r = 0; r < side; r++) for (var c = 0; c < side; c++) {
      var v = M.clamp(img[r * side + c], 0, 1);
      ctx.fillStyle = M.css([250 - v * 222, 248 - v * 206, 244 - v * 176]);
      ctx.fillRect(x + c * px, y + r * px, Math.ceil(px), Math.ceil(px));
    }
    ctx.strokeStyle = frame || C.border; ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, sz - 1, sz - 1);
  }

  var barGeom = null;
  function drawLeft() {
    var ctx = left.ctx, s = left;
    if (!s.w) return;
    M.clear(ctx, s);
    var pad = 10 * s.sc, nShow = 6, gap = 6 * s.sc;
    var sz = Math.min((s.w - 2 * pad - gap * (nShow - 1)) / nShow, (s.h * 0.5 - 3 * gap) / 2);
    var x0 = (s.w - (sz * nShow + gap * (nShow - 1))) / 2;
    for (var i = 0; i < nShow; i++) {
      var n = (i * 37 + 5) % X.length, x = x0 + i * (sz + gap);
      drawImg(ctx, X[n], x, pad, sz);
      drawImg(ctx, decode(encode(X[n])), x, pad + sz + gap, sz, M.rgba(C.orange, 0.6));
    }
    // bottom: code bars -> imagined image
    var top = pad + 2 * sz + 3 * gap + 6 * s.sc, bh = s.h - top - pad, bigSz = Math.min(bh, s.w * 0.3);
    var bx0 = pad + 4 * s.sc, bw = s.w - bigSz - 3 * pad - 30 * s.sc;
    ctx.font = M.font(s, 9, 500); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('CODE (' + K + (K === 1 ? ' NUMBER' : ' NUMBERS') + ')', bx0, top - 2 * s.sc);
    var bt = top + 14 * s.sc, bb = s.h - pad, mid = (bt + bb) / 2, half = (bb - bt) / 2;
    var slot = bw / K, barW = Math.min(26 * s.sc, slot * 0.7);
    ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(bx0, mid + 0.5); ctx.lineTo(bx0 + bw, mid + 0.5); ctx.stroke();
    var lim = codeLimit();
    for (var k = 0; k < K; k++) {
      var cxk = bx0 + slot * (k + 0.5), hgt = M.clamp(code[k] / lim, -1, 1) * half;
      ctx.fillStyle = k === dragBar ? C.orange : M.rgba(C.orange, 0.7);
      ctx.fillRect(cxk - barW / 2, hgt > 0 ? mid - hgt : mid, barW, Math.abs(hgt) || 1);
    }
    barGeom = { x0: bx0, slot: slot, mid: mid, half: half, top: bt, bottom: bb, w: bw, lim: lim };
    var ix = s.w - pad - bigSz, iy = top + (bh - bigSz) / 2;
    M.arrow(ctx, bx0 + bw + 6 * s.sc, mid, ix - 6 * s.sc, mid, C.muted, 1.4 * s.sc, 7 * s.sc);
    drawImg(ctx, decode(code), ix, iy, bigSz, C.orange);
    ctx.font = M.font(s, 9, 500); ctx.fillStyle = C.orange; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(dragBar >= 0 ? 'IMAGINED' : 'DECODED', ix + bigSz / 2, iy - 3 * s.sc);
  }
  function codeLimit() {
    var m = 1;
    codes.forEach(function (c) { for (var k = 0; k < c.length; k++) m = Math.max(m, Math.abs(c[k])); });
    return m * 1.1;
  }

  var vR = M.view(right, { x: [-1, 1], y: [-1, 1], pad: { l: 14, r: 14, t: 12, b: 14 } });
  function drawRight() {
    var ctx = right.ctx, s = right;
    if (!s.w) return;
    M.clear(ctx, s);
    var pts = codes.map(function (c) { return [c[0], K > 1 ? c[1] : 0]; });
    var bb = M.bounds(pts.concat([[code[0], K > 1 ? code[1] : 0]]), 0.12);
    if (K === 1) bb.y = [-1, 1];
    vR.setDomain(bb.x, bb.y);
    ctx.save(); vR.clip(ctx);
    pts.forEach(function (p, i) {
      var y = K > 1 ? p[1] : ((i * 0.6180339) % 1) * 1.6 - 0.8;
      M.dot(ctx, vR.px(p[0]), vR.py(y), 3 * s.sc, colorOf(i), null);
    });
    ctx.globalAlpha = 1;
    M.star(ctx, vR.px(code[0]), vR.py(K > 1 ? code[1] : 0), 8 * s.sc, C.orange, '#fff');
    ctx.restore();
    if (K === 1) { ctx.font = M.font(s, 9); ctx.fillStyle = C.muted; ctx.textAlign = 'center'; ctx.fillText('one number: only left-right means anything', s.w / 2, s.h - 6 * s.sc); }
    // loss sparkline
    if (losses.length > 2) {
      var w = 120 * s.sc, h = 40 * s.sc, x0 = s.w - w - 8 * s.sc, y0 = 8 * s.sc;
      var lo = Math.log(Math.min.apply(null, losses)), hi = Math.log(Math.max.apply(null, losses));
      ctx.fillStyle = M.rgba(C.surface, 0.94); ctx.strokeStyle = C.border;
      ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.fill(); ctx.stroke();
      ctx.font = M.font(s, 8.5, 500); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('COPY ERROR', x0 + 6 * s.sc, y0 + 4 * s.sc);
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.4 * s.sc; ctx.beginPath();
      losses.forEach(function (l, i) {
        var px = x0 + 6 * s.sc + i / (losses.length - 1) * (w - 12 * s.sc), py = y0 + h - 5 * s.sc - (Math.log(l) - lo) / ((hi - lo) || 1) * (h - 20 * s.sc);
        ctx[i ? 'lineTo' : 'moveTo'](px, py);
      });
      ctx.stroke();
    }
  }

  function draw() {
    if (!P) return;
    drawArchitecture(); drawLeft(); drawRight();
    var err = losses.length ? losses[losses.length - 1] : null;
    var verdict = !losses.length ? '<span class="k">untrained weights: press train to compress and reconstruct</span>'
      : err > 0.03 ? '<span class="o">fitting the coarse spatial structure</span>'
      : err > 0.012 ? '<span class="b">refining the bottleneck code</span>' : '<span class="g">reconstructed from ' + K + (K === 1 ? ' bottleneck unit' : ' bottleneck units') + '</span>';
    read.innerHTML = '<span class="k">' + D + ' pixels squeezed into</span> ' + K + ' <span class="k">(' + M.f(D / K, 0) + '&times; smaller)</span>' +
      '<span class="sep">&middot;</span><span class="k">passes through the data</span> ' + epoch +
      (err != null ? '<span class="sep">&middot;</span><span class="k">copy error</span> ' + M.f(err, 4) : '') +
      '<span class="sep">&middot;</span>' + verdict;
  }

  M.pointer(left.cv, {
    down: function (p) {
      if (!barGeom || p.y < barGeom.top - 6 || p.y > barGeom.bottom + 6 || p.x < barGeom.x0 || p.x > barGeom.x0 + barGeom.w) return false;
      dragBar = M.clamp(Math.floor((p.x - barGeom.x0) / barGeom.slot), 0, K - 1);
      code[dragBar] = M.clamp((barGeom.mid - p.y) / barGeom.half, -1, 1) * barGeom.lim; draw();
    },
    drag: function (p) { if (dragBar < 0) return; code[dragBar] = M.clamp((barGeom.mid - p.y) / barGeom.half, -1, 1) * barGeom.lim; draw(); },
    up: function () { dragBar = -1; draw(); },
    hover: function (p) { left.cv.style.cursor = barGeom && p.y > barGeom.top - 6 && p.x < barGeom.x0 + barGeom.w ? 'ns-resize' : ''; }
  });

  load();
  init();
  left.draw = draw; right.draw = draw;
  left.request();
  return { destroy: function () { loop.stop(); } };
});
