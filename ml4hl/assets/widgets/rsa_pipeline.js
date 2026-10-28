/* ══════════════════════════════════════════════════════════════════
   Alternative to Figure 13 — End-to-end RSA construction pipeline
   Walks through the 4 stages of Representational Similarity Analysis:
   (1) Item x Feature matrices -> (2) Pairwise RDMs -> (3) Flattened
   upper-triangle dissimilarity vectors -> (4) Second-order Spearman
   rank correlation across candidate models.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('rsa_pipeline', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'End-to-end RSA pipeline: from feature tables to upper triangles to model comparison',
    hint: 'Hover a pair in the RDM or vector barcode &middot; switch candidate model &middot; change distance metric'
  });

  var real = cfg.data && cfg.data.A;
  var seed = cfg.seed || 10;
  var N = 12, names, groups, A, models, activeModel = 'model_aligned', hoverPair = 0;
  var rdmA, rdmM, triA, triM, pairs, scores;

  function load() {
    var rng = M.rng(seed);
    if (real && cfg.data.A.length >= 8) {
      var step = Math.max(1, Math.floor(cfg.data.A.length / N));
      var idx = M.range(N).map(function (i) { return Math.min(cfg.data.A.length - 1, i * step); });
      names = idx.map(function (i) { return cfg.data.names[i]; });
      groups = idx.map(function (i) { return (cfg.data.groups || [])[i] || 'item'; });
      A = idx.map(function (i) { return cfg.data.A[i].slice(0, 16); });
      var B = idx.map(function (i) { return cfg.data.B[i].slice(0, 16); });
      models = {
        model_aligned: { label: cfg.data.labelB || 'neural model', X: B },
        category_rule: { label: 'category template', X: groups.map(function (g) { return ['animal', 'food', 'tool', 'vehicle'].map(function (c) { return g === c ? 1 : 0; }); }) },
        shuffled_null: { label: 'shuffled baseline', X: rng.shuffle(B.slice()) }
      };
    } else {
      names = ['cat', 'dog', 'owl', 'carp', 'apple', 'bread', 'grape', 'soup', 'saw', 'drill', 'plier', 'axe'];
      groups = ['animal', 'animal', 'animal', 'animal', 'food', 'food', 'food', 'food', 'tool', 'tool', 'tool', 'tool'];
      var cats = ['animal', 'food', 'tool'];
      var proto = cats.map(function () { return M.range(8).map(function () { return rng.normal(0, 1.1); }); });
      A = groups.map(function (g) {
        var c = proto[cats.indexOf(g)];
        return c.map(function (v) { return v + rng.normal(0, 0.35); });
      });
      var W = M.range(10).map(function () { return M.range(8).map(function () { return rng.normal(0, 0.5); }); });
      models = {
        model_aligned: {
          label: 'deep embedding (10-d)',
          X: A.map(function (a) { return W.map(function (w) { var s = 0; for (var j = 0; j < 8; j++) s += w[j] * a[j]; return Math.tanh(s) + rng.normal(0, 0.25); }); })
        },
        category_rule: {
          label: 'category indicator (3-d)',
          X: groups.map(function (g) { return cats.map(function (c) { return g === c ? 1 : 0; }); })
        },
        shuffled_null: {
          label: 'random features (10-d)',
          X: M.range(N).map(function () { return M.range(10).map(function () { return rng.normal(0, 1); }); })
        }
      };
    }
    compute();
  }

  function dissim(u, v, metric) {
    if (metric === 'euclidean') return M.dist(u, v);
    if (metric === 'cosine') {
      var d = 0, nu = 0, nv = 0;
      for (var i = 0; i < u.length; i++) { d += u[i] * v[i]; nu += u[i] * u[i]; nv += v[i] * v[i]; }
      return 1 - d / Math.sqrt((nu * nv) || 1);
    }
    return 1 - M.pearson(u, v);
  }

  function makeRDM(X, metric) {
    var mat = [], tri = [];
    for (var i = 0; i < N; i++) mat.push(new Float64Array(N));
    for (i = 0; i < N; i++) for (var j = i + 1; j < N; j++) {
      var d = dissim(X[i], X[j], metric);
      mat[i][j] = mat[j][i] = d;
      tri.push(d);
    }
    var r = M.rank(tri), m = tri.length;
    var rNorm = r.map(function (v) { return (v - 1) / Math.max(1, m - 1); });
    var k = 0, rMat = [];
    for (i = 0; i < N; i++) rMat.push(new Float64Array(N));
    for (i = 0; i < N; i++) for (j = i + 1; j < N; j++) {
      rMat[i][j] = rMat[j][i] = rNorm[k++];
    }
    return { mat: mat, rMat: rMat, tri: rNorm };
  }

  function compute() {
    var metric = metricT ? metricT.get() : (cfg.metric || 'correlation');
    pairs = [];
    for (var i = 0; i < N; i++) for (var j = i + 1; j < N; j++) pairs.push([i, j]);
    var ra = makeRDM(A, metric);
    rdmA = ra.rMat; triA = ra.tri;
    scores = {};
    Object.keys(models).forEach(function (k) {
      var rm = makeRDM(models[k].X, metric);
      scores[k] = { rdm: rm.rMat, tri: rm.tri, rho: M.pearson(triA, rm.tri) };
    });
    rdmM = scores[activeModel].rdm;
    triM = scores[activeModel].tri;
  }

  /* ─── DOM ─── */
  var stage = M.stage(root, { aspect: 0.42, minH: 250, maxH: 360, label: 'Four-stage RSA diagram: features, RDMs with highlighted upper triangles, flattened pair vectors, and model comparison bar chart' });
  var row = M.row(root);
  var modT = M.toggle(row, {
    label: 'Candidate B', value: activeModel,
    options: [
      ['model_aligned', 'neural embedding'],
      ['category_rule', 'category rule'],
      ['shuffled_null', 'shuffled control']
    ],
    onChange: function (v) { activeModel = v; compute(); draw(); }
  });
  var metricT = M.toggle(row, {
    label: 'Distance', value: cfg.metric || 'correlation',
    options: [['correlation', '1 - r'], ['cosine', 'cosine'], ['euclidean', 'euclidean']],
    onChange: function () { compute(); draw(); }
  });
  M.spacer(row);
  M.resetButton(row, function () { activeModel = 'model_aligned'; modT.set(activeModel, true); metricT.set(cfg.metric || 'correlation', true); hoverPair = 0; load(); draw(); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var hitBoxes = [];
  function draw() {
    var ctx = stage.ctx, s = stage;
    if (!s.w || !rdmA) return;
    M.clear(ctx, s);
    hitBoxes = [];

    var W = s.w, H = s.h, sc = s.sc, top = 26 * sc, hAvail = H - top - 16 * sc;
    /* 4 columns across the canvas: 1. Features, 2. RDMs (upper triangle), 3. Vector barcodes, 4. Second-order bar chart */
    var colX = [10 * sc, W * 0.23, W * 0.52, W * 0.72];
    var colW = [W * 0.19, W * 0.25, W * 0.16, W * 0.26];
    var headers = ['1. FEATURE MATRICES', '2. PAIRWISE RDMs (UPPER \u25b3)', '3. FLATTENED PAIRS', '4. 2ND-ORDER \u03c1'];
    ctx.font = M.font(s, 8.5, 600); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    headers.forEach(function (hd, i) { ctx.fillText(hd, colX[i], 7 * sc); });

    var hp = pairs[hoverPair] || [0, 1];

    /* Stage 1: two feature matrices (different widths!) */
    var matH = Math.min(hAvail * 0.43, 115 * sc);
    [[A, 'System A (' + A[0].length + 'd)', 0, C.blue], [models[activeModel].X, models[activeModel].label, 1, C.orange]].forEach(function (item) {
      var Xm = item[0], lbl = item[1], idx = item[2], col = item[3];
      var y0 = top + idx * (matH + 22 * sc), d = Math.min(12, Xm[0].length);
      var cw = Math.min(11 * sc, (colW[0] - 12 * sc) / d), rh = matH / N;
      ctx.font = M.font(s, 8, 500); ctx.fillStyle = col;
      ctx.fillText(lbl, colX[0], y0 - 11 * sc);
      for (var r = 0; r < N; r++) for (var c = 0; c < d; c++) {
        var v = M.clamp((Xm[r][c] + 2) / 4, 0, 1);
        ctx.fillStyle = M.css(M.diverge(v));
        ctx.fillRect(colX[0] + c * cw, y0 + r * rh, Math.ceil(cw - 0.5), Math.ceil(rh - 0.5));
      }
      [hp[0], hp[1]].forEach(function (r) {
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.5;
        ctx.strokeRect(colX[0], y0 + r * rh, d * cw, rh);
      });
    });

    /* Stage 2: Two RDMs with highlighted upper triangles */
    var rdmSz = Math.min(colW[1] * 0.88, matH), cell = rdmSz / N;
    [[rdmA, 'RDM A (' + N + '\u00d7' + N + ')', 0, C.blue], [rdmM, 'RDM B (' + N + '\u00d7' + N + ')', 1, C.orange]].forEach(function (item) {
      var R = item[0], lbl = item[1], idx = item[2], col = item[3];
      var x0 = colX[1] + 4 * sc, y0 = top + idx * (matH + 22 * sc);
      ctx.font = M.font(s, 8, 500); ctx.fillStyle = col;
      ctx.fillText(lbl, x0, y0 - 11 * sc);
      for (var r = 0; r < N; r++) for (var c = 0; c < N; c++) {
        var upper = c > r;
        ctx.fillStyle = r === c ? '#fff' : M.css(M.seq(0.08 + 0.92 * R[r][c]), upper ? 1 : 0.25);
        ctx.fillRect(x0 + c * cell, y0 + r * cell, Math.ceil(cell), Math.ceil(cell));
        if (upper) hitBoxes.push({ x: x0 + c * cell, y: y0 + r * cell, w: cell, h: cell, pair: [r, c] });
      }
      /* Upper-triangle outline */
      ctx.strokeStyle = col; ctx.lineWidth = 1.3 * sc;
      ctx.beginPath();
      ctx.moveTo(x0 + cell, y0); ctx.lineTo(x0 + N * cell, y0); ctx.lineTo(x0 + N * cell, y0 + (N - 1) * cell); ctx.closePath();
      ctx.stroke();
      ctx.strokeStyle = C.red; ctx.lineWidth = 2;
      ctx.strokeRect(x0 + hp[1] * cell, y0 + hp[0] * cell, cell, cell);
    });

    /* Stage 3: Two flattened upper-triangle vectors side by side (66 pairs) */
    var nP = pairs.length, vecH = hAvail - 10 * sc, pH = vecH / nP, bw = 20 * sc;
    var vxA = colX[2] + 10 * sc, vxB = vxA + bw + 18 * sc, vy0 = top + 6 * sc;
    ctx.font = M.font(s, 8, 600);
    ctx.fillStyle = C.blue; ctx.fillText('vec A', vxA, vy0 - 11 * sc);
    ctx.fillStyle = C.orange; ctx.fillText('vec B', vxB, vy0 - 11 * sc);
    for (var k = 0; k < nP; k++) {
      var py = vy0 + k * pH;
      ctx.fillStyle = M.css(M.seq(0.08 + 0.92 * triA[k]));
      ctx.fillRect(vxA, py, bw, Math.ceil(pH));
      ctx.fillStyle = M.css(M.seq(0.08 + 0.92 * triM[k]));
      ctx.fillRect(vxB, py, bw, Math.ceil(pH));
      hitBoxes.push({ x: vxA, y: py, w: vxB + bw - vxA, h: Math.max(2, pH), k: k });
    }
    var hy = vy0 + hoverPair * pH;
    ctx.strokeStyle = C.red; ctx.lineWidth = 1.8;
    ctx.strokeRect(vxA - 1, hy - 1, vxB + bw - vxA + 2, Math.max(3, pH + 2));

    /* Stage 4: Second-order Spearman rho bar chart across the 3 candidate models */
    var bx0 = colX[3] + 6 * sc, bwMax = colW[3] - 24 * sc, by0 = top + 12 * sc;
    var keys = ['model_aligned', 'category_rule', 'shuffled_null'];
    keys.forEach(function (key, idx) {
      var val = scores[key].rho, y = by0 + idx * 58 * sc, on = key === activeModel;
      ctx.font = M.font(s, 8.5, on ? 600 : 400);
      ctx.fillStyle = on ? C.ink : C.muted;
      ctx.fillText(models[key].label, bx0, y);
      var barY = y + 14 * sc, barH = 16 * sc;
      ctx.fillStyle = C.grid;
      ctx.fillRect(bx0, barY, bwMax, barH);
      var wBar = Math.max(2, Math.max(0, val) * bwMax);
      ctx.fillStyle = on ? C.green : M.rgba(C.blue, 0.55);
      ctx.fillRect(bx0, barY, wBar, barH);
      ctx.font = M.font(s, 9, 600); ctx.fillStyle = C.ink;
      ctx.fillText('\u03c1 = ' + M.f(val, 2), bx0 + wBar + 5 * sc, barY + 3 * sc);
    });

    var rhoNow = scores[activeModel].rho;
    read.innerHTML = '<span class="k">' + N + ' items &rarr; ' + pairs.length + ' upper-triangle pairs</span>' +
      '<span class="sep">&middot;</span><span class="r">pair (' + M.esc(names[hp[0]]) + ', ' + M.esc(names[hp[1]]) + ')</span>' +
      '<span class="sep">&middot;</span><span class="k">2nd-order Spearman \u03c1</span> <span class="g">' + M.f(rhoNow, 2) + '</span>' +
      '<span class="sep">&middot;</span><span class="b">different feature counts (' + A[0].length + 'd vs ' + models[activeModel].X[0].length + 'd) yield identical-sized ' + pairs.length + '-pair vectors</span>';
  }

  M.pointer(stage.cv, {
    down: function () { return false; },
    hover: function (p) {
      for (var i = 0; i < hitBoxes.length; i++) {
        var b = hitBoxes[i];
        if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
          var k = b.k != null ? b.k : pairs.findIndex(function (pr) { return pr[0] === b.pair[0] && pr[1] === b.pair[1]; });
          if (k >= 0 && k !== hoverPair) { hoverPair = k; draw(); }
          stage.cv.style.cursor = 'pointer';
          return;
        }
      }
      stage.cv.style.cursor = '';
    }
  });

  load();
  stage.draw = draw;
  stage.request();
});
