/* ══════════════════════════════════════════════════════════════════
   Figure 7 — One dataset, five classifiers
   Decision surfaces are fit in Python (ml4hl.zoo.decision_grids) for
   every model across its main setting, then shipped here as compact
   base64 grids: class index + confidence per cell.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('classifier_zoo', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'One dataset, five classifiers',
    hint: 'Pick a model &middot; slide its main setting &middot; watch train vs test accuracy'
  });

  var sets = cfg.data ? (Array.isArray(cfg.data) ? cfg.data : [cfg.data]) : [];
  if (!sets.length) {
    root.appendChild(M.h('div', { class: 'wnote', html: 'This figure needs data: <code>ml4hl.show("classifier_zoo", data=ml4hl.zoo.decision_grids(X, y))</code>' }));
    return;
  }
  var SHORT = { logistic: 'logistic', knn: 'k-NN', svm: 'SVM', tree: 'tree', gbm: 'boosting' };
  var ds = 0, model = cfg.model || 'knn', idx = cfg.setting == null ? 2 : cfg.setting, showTest = true;
  var cache = {};

  function decode(b64, res) {
    var key = b64.length + ':' + b64.slice(0, 32) + b64.slice(-32);
    if (cache[key]) return cache[key];
    var bin = atob(b64), n = res * res, off = document.createElement('canvas');
    off.width = res; off.height = res;
    var octx = off.getContext('2d'), img = octx.createImageData(res, res), cls = new Uint8Array(n);
    for (var k = 0; k < n; k++) {
      var c = bin.charCodeAt(2 * k), conf = bin.charCodeAt(2 * k + 1) / 255, col = M.hex(M.catColor(c));
      cls[k] = c;
      var mix = 0.18 + 0.5 * conf;
      img.data[4 * k] = Math.round(250 + (col[0] - 250) * mix);
      img.data[4 * k + 1] = Math.round(248 + (col[1] - 248) * mix);
      img.data[4 * k + 2] = Math.round(244 + (col[2] - 244) * mix);
      img.data[4 * k + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    return (cache[key] = { canvas: off, cls: cls });
  }

  /* ─── DOM ─── */
  var row0 = M.row(root), dsT = null;
  if (sets.length > 1) dsT = M.toggle(row0, { label: 'Dataset', value: 0, options: sets.map(function (d, i) { return [i, M.esc(d.name)]; }), onChange: function (v) { ds = v; syncModels(); drawAll(); } });
  var modelT = M.toggle(row0, {
    label: 'Model', value: model,
    options: Object.keys(SHORT).map(function (k) { return [k, SHORT[k]]; }),
    onChange: function (v) { model = v; syncSlider(); drawAll(); }
  });
  var split = M.split(root, 1.65);
  var mapTitle = M.mini(split.a, '');
  M.mini(split.b, 'accuracy at each setting');
  var main = M.stage(split.a, { height: function (w) { return M.clamp(w * 0.66, 240, 400); }, ref: 520, label: 'Decision regions of the selected classifier' });
  var side = M.stage(split.b, { height: function () { return main.h || 280; }, ref: 300, label: 'Training and test accuracy across settings' });
  var legend = M.legend(root, []);
  var row = M.row(root);
  var slider = M.slider(row, { label: 'Setting', min: 0, max: 4, step: 1, value: idx, accent: 'blue', fmt: function (i) { return settingLabel(i); }, onInput: function (v) { idx = v; drawAll(); } });
  var testBtn = M.button(row, 'test points', function () { showTest = !showTest; testBtn.classList.toggle('on', showTest); drawAll(); }, { aria: 'Toggle held-out test points' });
  testBtn.classList.add('on');
  M.spacer(row);
  M.resetButton(row, function () { model = cfg.model || 'knn'; modelT.set(model, true); idx = cfg.setting == null ? 2 : cfg.setting; syncSlider(); showTest = true; testBtn.classList.add('on'); drawAll(); });
  var read = M.readout(root);

  function D() { return sets[ds]; }
  function Mo() { return D().models[model] || D().models[Object.keys(D().models)[0]]; }
  function settingLabel(i) { var m = Mo(); return m ? m.param.split(' (')[0] + ' = ' + m.values[i] : String(i); }
  function syncSlider() { var m = Mo(); slider.input.max = m.values.length - 1; idx = Math.min(idx, m.values.length - 1); slider.set(idx, true); }
  function syncModels() {
    modelT.buttons.forEach(function (b) { b[1].style.display = D().models[b[0]] ? '' : 'none'; });
    if (!D().models[model]) { model = Object.keys(D().models)[0]; modelT.set(model, true); }
    syncSlider();
  }

  /* ─── Drawing ─── */
  var v = M.view(main, { x: [0, 1], y: [0, 1], pad: { l: 52, b: 40, t: 10, r: 10 } });
  var vs = M.view(side, { x: [-0.4, 4.4], y: [0, 1], pad: { l: 40, b: 38, t: 12, r: 10 } });

  function drawMain() {
    var ctx = main.ctx, s = main, d = D(), m = Mo();
    if (!s.w) return;
    v.setDomain(d.x, d.y);
    M.clear(ctx, s);
    var g = decode(m.grids[idx], d.res);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(g.canvas, v.L, v.T, v.R - v.L, v.B - v.T);
    ctx.restore();
    M.axes(ctx, v, s, { grid: false, xlabel: d.xlabel, ylabel: d.ylabel });
    ctx.save(); v.clip(ctx);
    for (var i = 0; i < d.X.length; i++) {
      var isTest = d.test[i] === 1;
      if (isTest && !showTest) continue;
      var px = v.px(d.X[i][0]), py = v.py(d.X[i][1]);
      var gi = M.clamp(Math.floor((d.X[i][0] - d.x[0]) / (d.x[1] - d.x[0]) * d.res), 0, d.res - 1);
      var gj = M.clamp(Math.floor((d.y[1] - d.X[i][1]) / (d.y[1] - d.y[0]) * d.res), 0, d.res - 1);
      var wrong = g.cls[gj * d.res + gi] !== d.labels[i], col = M.catColor(d.labels[i]);
      if (isTest) M.dot(ctx, px, py, 3.8 * s.sc, '#fff', wrong ? C.red : col, (wrong ? 2.2 : 1.8) * s.sc);
      else M.dot(ctx, px, py, 3.4 * s.sc, col, wrong ? C.red : '#fff', wrong ? 1.8 * s.sc : 0.9);
    }
    ctx.restore();
  }

  function drawSide() {
    var ctx = side.ctx, s = side, m = Mo();
    if (!s.w) return;
    var n = m.values.length, lo = Math.min.apply(null, m.test.concat(m.train));
    vs.setDomain([-0.4, n - 0.6], [Math.max(0, Math.floor((lo - 0.06) * 10) / 10), 1.005]);
    M.clear(ctx, s);
    M.axes(ctx, vs, s, { xticks: M.range(n), xfmt: function (t) { return String(m.values[t]); }, yfmt: function (t) { return Math.round(t * 100) + '%'; }, ny: 4, xlabel: m.param.split(' (')[0] });
    ctx.fillStyle = M.rgba(C.blue, 0.08);
    ctx.fillRect(vs.px(idx - 0.4), vs.T, vs.px(0.8) - vs.px(0), vs.B - vs.T);
    [['train', C.blue, true], ['test', C.orange, false]].forEach(function (sr) {
      ctx.strokeStyle = sr[1]; ctx.lineWidth = 1.8 * s.sc;
      ctx.beginPath();
      m[sr[0]].forEach(function (a, k) { ctx[k ? 'lineTo' : 'moveTo'](vs.px(k), vs.py(a)); });
      ctx.stroke();
      m[sr[0]].forEach(function (a, k) { M.dot(ctx, vs.px(k), vs.py(a), (k === idx ? 4.6 : 3.2) * s.sc, sr[2] ? sr[1] : '#fff', sr[1], 1.6 * s.sc); });
    });
  }

  function drawAll() {
    var d = D(), m = Mo();
    mapTitle.innerHTML = M.esc(m.label) + ' &middot; ' + M.esc(settingLabel(idx));
    drawMain(); drawSide();
    M.setLegend(legend, d.classes.map(function (c, i) { return { color: M.catColor(i), label: M.esc(c) }; }).concat([
      { color: C.ink, label: 'filled = training point' }, { color: C.ink, ring: true, label: 'hollow = test point' }, { color: C.red, ring: true, label: 'misclassified' }
    ]));
    var tr = m.train[idx], te = m.test[idx], gap = tr - te, bestTe = Math.max.apply(null, m.test);
    var verdict = gap > 0.07 && tr > 0.97
      ? '<span class="o">overfitting: near-perfect on training points, worse on new ones</span>'
      : te < bestTe - 0.05 && gap < 0.04
        ? '<span class="o">underfitting: too rigid to follow the categories</span>'
        : te >= bestTe - 0.01 ? '<span class="g">sweet spot: the best test accuracy for this model</span>'
        : '<span class="b">reasonable balance</span>';
    read.innerHTML = '<span class="k">train</span> <span class="b">' + Math.round(tr * 100) + '%</span>' +
      '<span class="sep">&middot;</span><span class="k">test</span> <span class="o">' + Math.round(te * 100) + '%</span>' +
      '<span class="sep">&middot;</span>' + verdict;
  }

  syncModels();
  main.draw = drawAll; side.draw = drawAll;
  main.onResize = function () { setTimeout(side.resize, 0); };
  main.request();
});
