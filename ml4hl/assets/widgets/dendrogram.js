/* ══════════════════════════════════════════════════════════════════
   Figure 10 — Hierarchical clustering: cut the tree
   Agglomerative clustering (Lance–Williams updates, same linkages as
   scipy / sklearn). Drag the cut line to decide how many clusters.
   ══════════════════════════════════════════════════════════════════ */
ML4HL.register('dendrogram', function (root, cfg) {
  var M = ML4HL, C = M.C;
  M.frame(root, cfg, {
    title: 'Hierarchical clustering: cut the tree',
    hint: 'Drag the dashed cut line &middot; hover a branch to see its members &middot; switch linkage'
  });

  var real = cfg.data && cfg.data.X;
  var seed = cfg.seed || 6, X, XY, labels, n, merges, order, nodeX, cut, hoverNode = -1, maxH;

  function load() {
    if (real) {
      X = cfg.data.X;
      XY = cfg.data.xy || (function () { var p = M.pca(X, 2); return p.Xc.map(function (r) { return [M.sum(r.map(function (v, j) { return v * p.components[0][j]; })), M.sum(r.map(function (v, j) { return v * p.components[1][j]; }))]; }); })();
      labels = cfg.data.labels || null;
    } else {
      var k = cfg.true_k || 3, d = M.sim.blobs(M.rng(seed), { k: k, n: Math.round((cfg.n || 34) / k), spread: 0.22 });
      X = d.X; XY = d.X; labels = null;
    }
    n = X.length;
  }

  function cluster(method) {
    var D = [];
    for (var i = 0; i < n; i++) { D.push(new Float64Array(n)); }
    for (i = 0; i < n; i++) for (var j = i + 1; j < n; j++) D[i][j] = D[j][i] = M.dist(X[i], X[j]);
    var active = M.range(n), id = M.range(n), size = new Array(n).fill(1), out = [];
    while (active.length > 1) {
      var bi = 0, bj = 1, bd = Infinity;
      for (var a = 0; a < active.length; a++) for (var b = a + 1; b < active.length; b++) {
        var d = D[active[a]][active[b]];
        if (d < bd) { bd = d; bi = a; bj = b; }
      }
      var p = active[bi], q = active[bj], sp = size[p], sq = size[q];
      out.push({ a: id[p], b: id[q], h: bd, size: sp + sq });
      for (var t = 0; t < active.length; t++) {
        var r = active[t];
        if (r === p || r === q) continue;
        var dpr = D[p][r], dqr = D[q][r], nd;
        if (method === 'single') nd = Math.min(dpr, dqr);
        else if (method === 'complete') nd = Math.max(dpr, dqr);
        else if (method === 'average') nd = (sp * dpr + sq * dqr) / (sp + sq);
        else {
          var sr = size[r], T = sp + sq + sr;
          nd = Math.sqrt(Math.max(0, ((sp + sr) * dpr * dpr + (sq + sr) * dqr * dqr - sr * bd * bd) / T));
        }
        D[p][r] = D[r][p] = nd;
      }
      size[p] = sp + sq; id[p] = n + out.length - 1;
      active.splice(bj, 1);
    }
    merges = out;
    maxH = out[out.length - 1].h;
    order = [];
    (function walk(node) { if (node < n) order.push(node); else { walk(merges[node - n].a); walk(merges[node - n].b); } })(2 * n - 2);
    nodeX = new Float64Array(2 * n - 1);
    order.forEach(function (leaf, i) { nodeX[leaf] = i; });
    merges.forEach(function (m, k) { nodeX[n + k] = (nodeX[m.a] + nodeX[m.b]) / 2; });
  }

  function heightOf(node) { return node < n ? 0 : merges[node - n].h; }
  function members(node) { var out = []; (function w(x) { if (x < n) out.push(x); else { w(merges[x - n].a); w(merges[x - n].b); } })(node); return out; }

  /* Clusters below the cut: maximal subtrees whose merge height is under it. */
  function assign() {
    var lab = new Int16Array(n).fill(-1), k = 0, nodeLab = new Int16Array(2 * n - 1).fill(-1);
    (function walk(node) {
      if (heightOf(node) <= cut) { members(node).forEach(function (m) { lab[m] = k; }); markDown(node, k); k++; return; }
      walk(merges[node - n].a); walk(merges[node - n].b);
    })(2 * n - 2);
    function markDown(node, c) { nodeLab[node] = c; if (node >= n) { markDown(merges[node - n].a, c); markDown(merges[node - n].b, c); } }
    return { lab: lab, k: k, nodeLab: nodeLab };
  }

  function cutFor(k) {
    var hs = merges.map(function (m) { return m.h; }).sort(function (a, b) { return a - b; });
    k = M.clamp(k, 1, n);
    if (k === 1) return maxH * 1.04;
    var hi = hs[n - k], lo = hs[n - k - 1] == null ? 0 : hs[n - k - 1];
    return (lo + hi) / 2;
  }
  function snapCut() {
    var hs = merges.map(function (m) { return m.h; }).sort(function (a, b) { return a - b; });
    for (var i = 0; i < hs.length; i++) if (hs[i] > cut) { cut = ((i ? hs[i - 1] : 0) + hs[i]) / 2; return; }
    cut = maxH * 1.04;
  }

  /* ─── DOM ─── */
  var split = M.split(root, 1);
  M.mini(split.a, real ? 'the data (2D view)' : 'the data');
  M.mini(split.b, 'the merge tree (height = distance when merged)');
  var hf = function (w) { return M.clamp(w * 0.85, 230, 360); };
  var sc = M.stage(split.a, { height: hf, ref: 420, label: 'Scatter plot colored by cluster' });
  var tree = M.stage(split.b, { height: hf, ref: 420, label: 'Dendrogram with a draggable cut line' });
  var row = M.row(root);
  var linkT = M.toggle(row, {
    label: 'Linkage', value: cfg.linkage || 'ward',
    options: [['single', 'single'], ['complete', 'complete'], ['average', 'average'], ['ward', 'Ward']],
    onChange: function (vv) { var k = assign().k; cluster(vv); cut = cutFor(k); drawAll(); }
  });
  M.spacer(row);
  if (!real) M.button(row, 'new data', function () { seed++; load(); cluster(linkT.get()); cut = cutFor(cfg.true_k || 3); drawAll(); }, { aria: 'Draw a new random dataset' });
  M.resetButton(row, function () { seed = cfg.seed || 6; load(); linkT.set(cfg.linkage || 'ward', true); cluster(linkT.get()); cut = cutFor(cfg.true_k || 3); drawAll(); });
  var read = M.readout(root);

  /* ─── Drawing ─── */
  var vS = M.view(sc, { x: [0, 1], y: [0, 1], equal: true, pad: { l: 12, r: 12, t: 12, b: 12 } });
  var showLabels = function () { return labels && n <= 44; };
  var vT = M.view(tree, { x: [0, 1], y: [0, 1], pad: { l: 40, r: 10, t: 12, b: 12 } });

  function drawScatter(asg) {
    var ctx = sc.ctx, s = sc;
    if (!s.w) return;
    var bb = M.bounds(XY, 0.12);
    vS.setDomain(bb.x, bb.y);
    M.clear(ctx, s);
    var hl = hoverNode >= 0 ? members(hoverNode) : [];
    var groups = {};
    for (var i = 0; i < n; i++) (groups[asg.lab[i]] = groups[asg.lab[i]] || []).push(XY[i]);
    Object.keys(groups).forEach(function (g) {
      var pts = groups[g];
      if (pts.length < 3) return;
      var hull = convexHull(pts);
      ctx.fillStyle = M.rgba(M.catColor(+g), 0.1); ctx.strokeStyle = M.rgba(M.catColor(+g), 0.45); ctx.lineWidth = 1.2 * s.sc;
      ctx.beginPath();
      hull.forEach(function (p, j) { ctx[j ? 'lineTo' : 'moveTo'](vS.px(p[0]), vS.py(p[1])); });
      ctx.closePath(); ctx.fill(); ctx.stroke();
    });
    for (i = 0; i < n; i++) {
      var on = hl.indexOf(i) >= 0;
      M.dot(ctx, vS.px(XY[i][0]), vS.py(XY[i][1]), (on ? 5 : 3.6) * s.sc, M.catColor(asg.lab[i]), on ? C.ink : '#fff', on ? 1.6 * s.sc : 0.9);
      if (labels && n <= 44) {
        ctx.font = M.font(s, 8); ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(String(labels[i]), vS.px(XY[i][0]) + 5 * s.sc, vS.py(XY[i][1]));
      }
    }
  }

  function convexHull(P) {
    var pts = P.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    function cross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
    var lo = [], up = [];
    pts.forEach(function (p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); });
    pts.slice().reverse().forEach(function (p) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); });
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }

  function drawTree(asg) {
    var ctx = tree.ctx, s = tree;
    if (!s.w) return;
    vT.pad.b = showLabels() ? 46 : 14;
    vT.setDomain([-0.6, n - 0.4], [0, maxH * 1.08]);
    M.clear(ctx, s);
    M.axes(ctx, vT, s, { xlabels: false, ylabel: 'merge distance', ny: 4, grid: 'y' });
    ctx.lineCap = 'round';
    merges.forEach(function (m, k) {
      var node = n + k, h = m.h, col = asg.nodeLab[node] >= 0 ? M.catColor(asg.nodeLab[node]) : '#9b968b';
      var hl = hoverNode >= 0 && members(hoverNode).length && isInside(node, hoverNode);
      ctx.strokeStyle = col; ctx.lineWidth = (hl ? 2.8 : 1.6) * s.sc;
      ctx.beginPath();
      [m.a, m.b].forEach(function (ch) { ctx.moveTo(vT.px(nodeX[ch]), vT.py(heightOf(ch))); ctx.lineTo(vT.px(nodeX[ch]), vT.py(h)); });
      ctx.moveTo(vT.px(nodeX[m.a]), vT.py(h)); ctx.lineTo(vT.px(nodeX[m.b]), vT.py(h));
      ctx.stroke();
    });
    if (showLabels()) {
      ctx.save(); ctx.font = M.font(s, 8); ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      order.forEach(function (leaf, i) {
        ctx.save(); ctx.translate(vT.px(i), vT.B + 4 * s.sc); ctx.rotate(-Math.PI / 2.4);
        ctx.fillText(String(labels[leaf]).slice(0, 14), 0, 0); ctx.restore();
      });
      ctx.restore();
    }
    var cy = vT.py(cut);
    ctx.strokeStyle = C.red; ctx.lineWidth = 1.6 * s.sc; ctx.setLineDash([6 * s.sc, 4 * s.sc]);
    ctx.beginPath(); ctx.moveTo(vT.L, cy); ctx.lineTo(vT.R, cy); ctx.stroke(); ctx.setLineDash([]);
    M.pill(ctx, s, asg.k + (asg.k === 1 ? ' cluster' : ' clusters'), vT.R - 4 * s.sc, cy - 12 * s.sc, { align: 'right', color: C.red, size: 9 });
  }
  function isInside(node, anc) { if (node === anc) return true; if (anc < n) return false; return isInside(node, merges[anc - n].a) || isInside(node, merges[anc - n].b); }

  function drawAll() {
    if (!merges) return;
    var asg = assign();
    drawScatter(asg); drawTree(asg);
    var note = { single: 'single linkage chains neighbors together (long, stringy clusters)',
      complete: 'complete linkage keeps every cluster compact',
      average: 'average linkage is a compromise between single and complete',
      ward: 'Ward merges whichever pair adds the least spread (round, even clusters)' }[linkT.get()];
    read.innerHTML = '<span class="k">cut at</span> ' + M.f(cut, 2) + '<span class="sep">&middot;</span><span class="b">' + asg.k + (asg.k === 1 ? ' cluster' : ' clusters') + '</span>' +
      (hoverNode >= 0 ? '<span class="sep">&middot;</span><span class="k">branch with</span> ' + members(hoverNode).length + ' members, merged at ' + M.f(heightOf(hoverNode), 2) : '') +
      '<span class="sep">&middot;</span><span class="k">' + note + '</span>';
  }

  var dragging = false;
  M.pointer(tree.cv, {
    down: function (p) { if (!vT.inside(p.x, p.y + 6)) return false; dragging = true; cut = M.clamp(vT.iy(p.y), 0, maxH * 1.07); drawAll(); },
    drag: function (p) { cut = M.clamp(vT.iy(p.y), 0, maxH * 1.07); drawAll(); },
    up: function () { dragging = false; snapCut(); drawAll(); },
    hover: function (p) {
      tree.cv.style.cursor = Math.abs(p.y - vT.py(cut)) < 8 ? 'ns-resize' : 'pointer';
      var best = -1, bd = 10 * tree.sc;
      merges.forEach(function (m, k) {
        var y = vT.py(m.h), x0 = vT.px(nodeX[m.a]), x1 = vT.px(nodeX[m.b]);
        if (p.x >= Math.min(x0, x1) - 4 && p.x <= Math.max(x0, x1) + 4 && Math.abs(p.y - y) < bd) { bd = Math.abs(p.y - y); best = n + k; }
      });
      if (best !== hoverNode) { hoverNode = best; drawAll(); }
    },
    leave: function () { if (hoverNode >= 0) { hoverNode = -1; drawAll(); } },
    key: function (e) {
      var k = assign().k;
      if (e.key === 'ArrowUp') { cut = cutFor(k - 1); drawAll(); return true; }
      if (e.key === 'ArrowDown') { cut = cutFor(k + 1); drawAll(); return true; }
    }
  });

  load();
  cluster(cfg.linkage || 'ward');
  cut = cutFor(real ? 3 : (cfg.true_k || 3));
  sc.draw = drawAll; tree.draw = drawAll;
  sc.request();
});
