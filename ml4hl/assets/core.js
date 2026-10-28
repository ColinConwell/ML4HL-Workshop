/* ══════════════════════════════════════════════════════════════════
   ML4HL widget runtime — shared helpers for every ML-4-HL widget.
   Vanilla JS, no dependencies. Safe to inline many times on one page:
   only the first copy installs itself.
   ══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';
  if (window.ML4HL && window.ML4HL.__core) return;
  var M = (window.ML4HL = window.ML4HL || {});
  M.__core = '0.1.0';
  M.registry = M.registry || {};
  M.queue = M.queue || [];

  /* ─── Registry ─────────────────────────────────────────────────── */

  M.register = function (name, fn) {
    M.registry[name] = fn;
    var waiting = M.queue.filter(function (j) { return j.name === name; });
    M.queue = M.queue.filter(function (j) { return j.name !== name; });
    waiting.forEach(function (j) { M.mount(j.name, j.target, j.cfg); });
  };

  M.mount = function (name, target, cfg) {
    var el = typeof target === 'string' ? document.getElementById(target) : target;
    if (!el) return null;
    var fn = M.registry[name];
    if (!fn) { M.queue.push({ name: name, target: el, cfg: cfg }); return null; }
    if (el.__ml4hl && el.__ml4hl.destroy) el.__ml4hl.destroy();
    el.innerHTML = '';
    el.classList.add('ml4hl', 'widget');
    el.setAttribute('data-widget', name);
    if (window.ResizeObserver && !el.__ml4hlRO) {
      el.__ml4hlRO = new ResizeObserver(function () { el.classList.toggle('narrow', el.clientWidth < 640); });
      el.__ml4hlRO.observe(el);
    }
    el.classList.toggle('narrow', el.clientWidth > 0 && el.clientWidth < 640);
    var api = null;
    try {
      api = fn(el, cfg || {}) || {};
    } catch (err) {
      console.error('[ML4HL] ' + name + ':', err);
      el.innerHTML = '<div class="ml4hl-error">Widget <b>' + M.esc(name) +
        '</b> failed to start: ' + M.esc(err && err.message) + '</div>';
      return null;
    }
    el.__ml4hl = api;
    M.colabResize();
    return api;
  };

  /* ─── Palette ──────────────────────────────────────────────────── */

  M.C = {
    bg: '#faf8f4', surface: '#fffdf9', ink: '#2a2a2a', muted: '#8a877e',
    faint: '#c9c4b8', grid: '#e7e3da', border: '#e2ddd2', codeBg: '#f3f0e8',
    blue: '#2563eb', blueDeep: '#1d4ed8', blueSoft: '#dbe6fd',
    orange: '#d97706', orangeDeep: '#a35c05', orangeSoft: '#fbe7c6',
    green: '#16a34a', greenSoft: '#bbf7d0', red: '#b83a2a', gold: '#ca8a04',
    purple: '#6a4c93', teal: '#0f766e'
  };
  M.cat = ['#2563eb', '#d97706', '#0f766e', '#b83a2a', '#6a4c93', '#ca8a04',
    '#db2777', '#0891b2', '#65a30d', '#4b5563', '#9333ea', '#ea580c'];
  if (window.ML4HLTheme) ML4HLTheme.apply();
  M.catColor = function (i) { return i < 0 ? M.C.faint : M.cat[i % M.cat.length]; };

  M.hex = function (hex) {
    var v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  };
  M.rgba = function (hex, a) {
    var c = M.hex(hex);
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
  };
  function ramp(stops, t) {
    t = Math.max(0, Math.min(1, t));
    var x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    var a = stops[i], b = stops[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  var MAGMA = [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
    [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]];
  var DIVERGE = [[37, 99, 235], [128, 164, 240], [250, 248, 244], [240, 177, 96], [217, 119, 6]];
  var SEQ = [[250, 248, 244], [219, 230, 253], [128, 164, 240], [37, 99, 235], [23, 37, 84]];
  M.magma = function (t) { return ramp(MAGMA, t); };
  M.diverge = function (t) { return ramp(DIVERGE, t); };
  M.seq = function (t) { return ramp(SEQ, t); };
  M.css = function (rgb, a) {
    return 'rgba(' + (rgb[0] | 0) + ',' + (rgb[1] | 0) + ',' + (rgb[2] | 0) + ',' + (a == null ? 1 : a) + ')';
  };

  /* ─── Small utilities ──────────────────────────────────────────── */

  M.esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };
  M.strip = function (s) { return String(s || '').replace(/<[^>]*>/g, '').replace(/&[a-z]+;/g, ' ').trim(); };
  M.clamp = function (v, a, b) { return v < a ? a : v > b ? b : v; };
  M.lerp = function (a, b, t) { return a + (b - a) * t; };
  M.ease = function (t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  M.f = function (v, d) {
    if (v == null || !isFinite(v)) return '\u2014';
    var s = Math.abs(v).toFixed(d == null ? 2 : d);
    return (v < 0 && +s !== 0 ? '\u2212' : '') + s;
  };
  M.pct = function (v, d) { return M.f(100 * v, d == null ? 0 : d) + '%'; };
  M.sum = function (a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; };
  M.mean = function (a) { return a.length ? M.sum(a) / a.length : NaN; };
  M.variance = function (a) {
    var m = M.mean(a), s = 0;
    for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return a.length > 1 ? s / (a.length - 1) : 0;
  };
  M.std = function (a) { return Math.sqrt(M.variance(a)); };
  M.quantile = function (a, q) {
    var s = a.slice().sort(function (x, y) { return x - y; });
    var p = (s.length - 1) * q, i = Math.floor(p), f = p - i;
    return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * f;
  };
  M.argmax = function (a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  M.argmin = function (a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] < a[b]) b = i; return b; };
  M.range = function (n) { var r = new Array(n); for (var i = 0; i < n; i++) r[i] = i; return r; };
  M.sigmoid = function (z) { return 1 / (1 + Math.exp(-z)); };
  M.dist2 = function (a, b) { var s = 0; for (var i = 0; i < a.length; i++) { var d = a[i] - b[i]; s += d * d; } return s; };
  M.dist = function (a, b) { return Math.sqrt(M.dist2(a, b)); };

  /* Seeded RNG (mulberry32) so simulations are reproducible. */
  M.rng = function (seed) {
    var a = (seed == null ? 1 : seed) >>> 0 || 1;
    var r = {
      next: function () {
        a = (a + 0x6D2B79F5) >>> 0;
        var t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      },
      uniform: function (lo, hi) { return lo + (hi - lo) * r.next(); },
      int: function (n) { return Math.floor(r.next() * n); },
      normal: function (m, s) {
        var u = 1 - r.next(), v = r.next();
        return (m || 0) + (s == null ? 1 : s) * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      },
      shuffle: function (arr) {
        for (var i = arr.length - 1; i > 0; i--) { var j = r.int(i + 1), t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
        return arr;
      },
      choice: function (arr) { return arr[r.int(arr.length)]; }
    };
    return r;
  };

  /* ─── Linear algebra & statistics ──────────────────────────────── */

  /* Solve A x = b by Gaussian elimination with partial pivoting. */
  M.solve = function (A, b) {
    var n = A.length, m = A.map(function (row, i) { return row.slice().concat([b[i]]); });
    for (var c = 0; c < n; c++) {
      var p = c;
      for (var r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
      var t = m[c]; m[c] = m[p]; m[p] = t;
      var piv = m[c][c] || 1e-12;
      for (r = c + 1; r < n; r++) {
        var f = m[r][c] / piv;
        if (!f) continue;
        for (var k = c; k <= n; k++) m[r][k] -= f * m[c][k];
      }
    }
    var x = new Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var s = m[i][n];
      for (var j = i + 1; j < n; j++) s -= m[i][j] * x[j];
      x[i] = s / (m[i][i] || 1e-12);
    }
    return x;
  };

  /* Least squares with optional ridge penalty: min |Phi w - y|^2 + lam |w|^2 */
  M.lstsq = function (Phi, y, lam) {
    var p = Phi[0].length, A = [], b = new Array(p).fill(0);
    for (var i = 0; i < p; i++) A.push(new Array(p).fill(0));
    for (var r = 0; r < Phi.length; r++) {
      var row = Phi[r];
      for (i = 0; i < p; i++) {
        b[i] += row[i] * y[r];
        for (var j = i; j < p; j++) A[i][j] += row[i] * row[j];
      }
    }
    for (i = 0; i < p; i++) { A[i][i] += lam || 0; for (j = 0; j < i; j++) A[i][j] = A[j][i]; }
    return M.solve(A, b);
  };

  M.linfit = function (x, y) {
    var mx = M.mean(x), my = M.mean(y), sxy = 0, sxx = 0;
    for (var i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    var m = sxx ? sxy / sxx : 0;
    return { m: m, b: my - m * mx };
  };

  /* Symmetric eigendecomposition (cyclic Jacobi). Returns values (desc) and
     vectors as an array of column vectors. Fine for n up to ~100. */
  M.eigh = function (S) {
    var n = S.length, A = S.map(function (r) { return r.slice(); }), V = [];
    for (var i = 0; i < n; i++) { V.push(new Array(n).fill(0)); V[i][i] = 1; }
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0;
      for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-18) break;
      for (p = 0; p < n; p++) {
        for (q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < 1e-14) continue;
          var theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
          var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (var k = 0; k < n; k++) {
            var akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
          }
          for (k = 0; k < n; k++) {
            var apk = A[p][k], aqk = A[q][k];
            A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
          }
          for (k = 0; k < n; k++) {
            var vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    var idx = M.range(n).sort(function (a, b) { return A[b][b] - A[a][a]; });
    return {
      values: idx.map(function (i) { return A[i][i]; }),
      vectors: idx.map(function (i) { return V.map(function (row) { return row[i]; }); })
    };
  };

  /* Principal components of rows X (n x d). Uses the smaller of the
     covariance (d x d) and Gram (n x n) problems. */
  M.pca = function (X, k) {
    var n = X.length, d = X[0].length, mu = new Array(d).fill(0), i, j, r;
    for (r = 0; r < n; r++) for (j = 0; j < d; j++) mu[j] += X[r][j] / n;
    var Xc = X.map(function (row) { return row.map(function (v, j) { return v - mu[j]; }); });
    var comps, values;
    if (d <= n) {
      var C = [];
      for (i = 0; i < d; i++) C.push(new Array(d).fill(0));
      for (r = 0; r < n; r++) for (i = 0; i < d; i++) { var xi = Xc[r][i]; if (!xi) continue; for (j = i; j < d; j++) C[i][j] += xi * Xc[r][j]; }
      for (i = 0; i < d; i++) for (j = 0; j < i; j++) C[i][j] = C[j][i];
      var e = M.eigh(C);
      values = e.values.map(function (v) { return v / Math.max(1, n - 1); });
      comps = e.vectors;
    } else {
      var G = [];
      for (i = 0; i < n; i++) { G.push(new Array(n).fill(0)); }
      for (i = 0; i < n; i++) for (j = i; j < n; j++) { var s = 0; for (var c = 0; c < d; c++) s += Xc[i][c] * Xc[j][c]; G[i][j] = G[j][i] = s; }
      var g = M.eigh(G);
      values = g.values.map(function (v) { return Math.max(0, v) / Math.max(1, n - 1); });
      comps = g.vectors.map(function (u, idx) {
        var w = new Array(d).fill(0), norm = 0;
        for (var rr = 0; rr < n; rr++) for (var cc = 0; cc < d; cc++) w[cc] += u[rr] * Xc[rr][cc];
        for (cc = 0; cc < d; cc++) norm += w[cc] * w[cc];
        norm = Math.sqrt(norm) || 1;
        return w.map(function (v) { return v / norm; });
      });
    }
    k = Math.min(k || d, comps.length);
    return { mean: mu, components: comps.slice(0, k), values: values, Xc: Xc };
  };

  M.rank = function (a) {
    var idx = M.range(a.length).sort(function (i, j) { return a[i] - a[j]; });
    var r = new Array(a.length), i = 0;
    while (i < idx.length) {
      var j = i;
      while (j + 1 < idx.length && a[idx[j + 1]] === a[idx[i]]) j++;
      for (var k = i; k <= j; k++) r[idx[k]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return r;
  };
  M.pearson = function (a, b) {
    var ma = M.mean(a), mb = M.mean(b), sab = 0, saa = 0, sbb = 0;
    for (var i = 0; i < a.length; i++) {
      var da = a[i] - ma, db = b[i] - mb;
      sab += da * db; saa += da * da; sbb += db * db;
    }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
  };
  M.spearman = function (a, b) { return M.pearson(M.rank(a), M.rank(b)); };

  /* ─── Simulated datasets ───────────────────────────────────────── */

  M.sim = {
    linear: function (rng, o) {
      o = o || {};
      var n = o.n || 40, x0 = o.x ? o.x[0] : -1, x1 = o.x ? o.x[1] : 3, xs = [], ys = [];
      for (var i = 0; i < n; i++) {
        var x = rng.uniform(x0, x1);
        xs.push(x); ys.push((o.slope == null ? 0.8 : o.slope) * x + (o.intercept == null ? 0.5 : o.intercept) + rng.normal(0, o.noise == null ? 0.5 : o.noise));
      }
      return { x: xs, y: ys };
    },
    blobs: function (rng, o) {
      o = o || {};
      var k = o.k || 2, n = o.n || 60, spread = o.spread == null ? 0.35 : o.spread, X = [], y = [];
      var centers = o.centers;
      if (!centers) {
        centers = [];
        var phase = rng.uniform(0, Math.PI * 2), rad = k === 2 ? 0.62 : 0.85;
        for (var c = 0; c < k; c++) {
          var a = phase + (2 * Math.PI * c) / k + rng.uniform(-0.25, 0.25);
          centers.push([rad * Math.cos(a) + rng.normal(0, 0.08), rad * Math.sin(a) + rng.normal(0, 0.08)]);
        }
      }
      for (c = 0; c < k; c++) {
        var sx = spread * rng.uniform(0.7, 1.3), sy = spread * rng.uniform(0.7, 1.3), rot = rng.uniform(0, Math.PI);
        for (var i = 0; i < n; i++) {
          var u = rng.normal(0, sx), v = rng.normal(0, sy);
          X.push([centers[c][0] + u * Math.cos(rot) - v * Math.sin(rot), centers[c][1] + u * Math.sin(rot) + v * Math.cos(rot)]);
          y.push(c);
        }
      }
      return { X: X, y: y, centers: centers };
    },
    moons: function (rng, o) {
      o = o || {};
      var n = o.n || 80, noise = o.noise == null ? 0.12 : o.noise, X = [], y = [];
      for (var i = 0; i < n; i++) {
        var t = Math.PI * rng.next();
        X.push([(Math.cos(t) - 0.5) * 0.9 + rng.normal(0, noise), (Math.sin(t) - 0.25) * 0.9 + rng.normal(0, noise)]); y.push(0);
        X.push([(1 - Math.cos(t) - 0.5) * 0.9 + rng.normal(0, noise), (0.5 - Math.sin(t) - 0.25) * 0.9 + rng.normal(0, noise)]); y.push(1);
      }
      return { X: X, y: y };
    },
    circles: function (rng, o) {
      o = o || {};
      var n = o.n || 80, noise = o.noise == null ? 0.08 : o.noise, X = [], y = [];
      for (var i = 0; i < n; i++) {
        var t = 2 * Math.PI * rng.next(), s = 2 * Math.PI * rng.next();
        X.push([1.05 * Math.cos(t) + rng.normal(0, noise), 1.05 * Math.sin(t) + rng.normal(0, noise)]); y.push(0);
        X.push([0.45 * Math.cos(s) + rng.normal(0, noise), 0.45 * Math.sin(s) + rng.normal(0, noise)]); y.push(1);
      }
      return { X: X, y: y };
    },
    xor: function (rng, o) {
      o = o || {};
      var n = o.n || 80, noise = o.noise == null ? 0.18 : o.noise, X = [], y = [];
      for (var i = 0; i < 2 * n; i++) {
        var qx = rng.next() < 0.5 ? -1 : 1, qy = rng.next() < 0.5 ? -1 : 1;
        X.push([qx * 0.55 + rng.normal(0, noise + 0.1), qy * 0.55 + rng.normal(0, noise + 0.1)]);
        y.push(qx * qy > 0 ? 0 : 1);
      }
      return { X: X, y: y };
    },
    spiral: function (rng, o) {
      o = o || {};
      var n = o.n || 80, k = o.k || 2, noise = o.noise == null ? 0.06 : o.noise, X = [], y = [];
      for (var c = 0; c < k; c++) {
        for (var i = 0; i < n; i++) {
          var t = (i / n) * 1.6 * Math.PI + 0.3, r = (0.12 + 0.95 * (i / n));
          var a = t + (2 * Math.PI * c) / k;
          X.push([r * Math.cos(a) + rng.normal(0, noise), r * Math.sin(a) + rng.normal(0, noise)]); y.push(c);
        }
      }
      return { X: X, y: y };
    }
  };

  /* Bounding box of 2D points, padded. */
  M.bounds = function (X, pad) {
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    X.forEach(function (p) { if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0]; if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; });
    var px = (x1 - x0 || 1) * (pad == null ? 0.08 : pad), py = (y1 - y0 || 1) * (pad == null ? 0.08 : pad);
    return { x: [x0 - px, x1 + px], y: [y0 - py, y1 + py] };
  };

  /* ─── DOM helpers ──────────────────────────────────────────────── */

  var h = (M.h = function (tag, attrs, kids) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v == null || v === false) return;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k === 'html') el.innerHTML = v;
        else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
        else if (k.slice(0, 2) === 'on' && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? '' : v);
      });
    }
    (kids || []).forEach(function (c) {
      if (c == null) return;
      el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return el;
  });

  /* Title row: "Figure N — Title" plus an interaction hint. */
  M.frame = function (root, cfg, d) {
    var fig = cfg.fig != null ? cfg.fig : d.fig;
    var title = cfg.title || d.title || '';
    var hint = cfg.hint != null ? cfg.hint : d.hint;
    root.appendChild(h('div', { class: 'widget-head' }, [
      h('div', { class: 'widget-title', html: (fig != null ? 'Figure ' + M.esc(fig) + ' &mdash; ' : '') + M.esc(title) }),
      hint ? h('div', { class: 'widget-hint', html: hint }) : null
    ]));
  };

  M.row = function (parent, cls) { return parent.appendChild(h('div', { class: 'wrow' + (cls ? ' ' + cls : '') })); };
  M.label = function (row, text) { return row.appendChild(h('span', { class: 'wlabel', html: text })); };
  M.spacer = function (row) { return row.appendChild(h('span', { class: 'wspacer' })); };
  M.readout = function (parent, cls) {
    return parent.appendChild(h('div', { class: 'readout wread' + (cls ? ' ' + cls : ''), 'aria-live': 'polite' }));
  };
  M.note = function (parent, html) { return parent.appendChild(h('div', { class: 'wnote', html: html })); };

  M.slider = function (row, o) {
    if (o.label) M.label(row, o.label);
    var toInput = o.log ? function (v) { return Math.log10(v); } : function (v) { return v; };
    var fromInput = o.log ? function (v) { return Math.pow(10, v); } : function (v) { return v; };
    var input = h('input', {
      type: 'range', min: toInput(o.min), max: toInput(o.max),
      step: o.log ? (toInput(o.max) - toInput(o.min)) / 200 : (o.step || 1),
      value: toInput(o.value), 'aria-label': o.aria || M.strip(o.label) || 'value',
      class: o.accent ? 'accent-' + o.accent : null
    });
    var out = h('span', { class: 'readout wval' + (o.accent ? ' accent-' + o.accent : '') });
    var fmt = o.fmt || function (v) { return String(v); };
    row.appendChild(input);
    if (o.show !== false) row.appendChild(out);
    var api = {
      input: input,
      get: function () { return fromInput(+input.value); },
      set: function (v, silent) {
        input.value = toInput(v);
        out.innerHTML = fmt(api.get());
        if (!silent && o.onInput) o.onInput(api.get());
      }
    };
    out.innerHTML = fmt(api.get());
    input.addEventListener('input', function () {
      out.innerHTML = fmt(api.get());
      if (o.onInput) o.onInput(api.get());
    });
    if (o.onChange) input.addEventListener('change', function () { o.onChange(api.get()); });
    return api;
  };

  M.toggle = function (row, o) {
    if (o.label) M.label(row, o.label);
    var grp = h('span', { class: 'wgroup', role: 'group', 'aria-label': o.aria || M.strip(o.label) || 'options' });
    var cur = o.value;
    var btns = o.options.map(function (opt) {
      var b = h('button', { type: 'button', class: 'wbtn', html: opt[1], 'aria-pressed': 'false' });
      b.addEventListener('click', function () { set(opt[0]); });
      grp.appendChild(b);
      return [opt[0], b];
    });
    function set(v, silent) {
      cur = v;
      btns.forEach(function (p) { p[1].classList.toggle('on', p[0] === v); p[1].setAttribute('aria-pressed', String(p[0] === v)); });
      if (!silent && o.onChange) o.onChange(v);
    }
    set(cur, true);
    row.appendChild(grp);
    return { get: function () { return cur; }, set: set, el: grp, buttons: btns };
  };

  M.button = function (row, html, onClick, o) {
    o = o || {};
    var b = h('button', {
      type: 'button', class: 'wbtn' + (o.cls ? ' ' + o.cls : ''), html: html,
      'aria-label': o.aria || null, 'data-role': o.role || null, title: o.title || null
    });
    b.addEventListener('click', onClick);
    row.appendChild(b);
    return b;
  };
  M.resetButton = function (row, onClick, text) {
    return M.button(row, text || 'Reset', onClick, { cls: 'wbtn-reset', role: 'reset' });
  };

  M.legend = function (parent, items) {
    var el = parent.appendChild(h('div', { class: 'wlegend' }));
    M.setLegend(el, items);
    return el;
  };
  M.setLegend = function (el, items) {
    el.innerHTML = '';
    items.forEach(function (it) {
      var key = Object.keys(M.C).find(function(k){return M.C[k] === it.color;});
      var color = key ? 'var(--ml-' + key + ')' : it.color;
      el.appendChild(h('span', { class: 'wchip' }, [
        h('i', { class: 'wdot' + (it.ring ? ' ring' : '') + (it.line ? ' line' : ''), style: { background: it.ring ? 'transparent' : color, borderColor: color } }),
        h('span', { html: it.label })
      ]));
    });;
  };

  /* ─── Canvas stage (DPR-aware, resize-observing) ───────────────── */

  M.stage = function (parent, o) {
    o = o || {};
    var wrap = h('div', { class: 'canvas-wrap ' + (o.cls || 'plain') });
    var cv = h('canvas', { role: 'img', 'aria-label': o.label || 'Interactive figure' });
    wrap.appendChild(cv);
    var ov = null;
    if (o.overlay) { ov = h('canvas', { class: 'overlay', 'aria-hidden': 'true' }); wrap.appendChild(ov); }
    parent.appendChild(wrap);
    var s = {
      wrap: wrap, cv: cv, ctx: cv.getContext('2d'), ov: ov, octx: ov ? ov.getContext('2d') : null,
      w: 0, h: 0, sc: 1, dpr: 1, draw: null, onResize: null
    };
    function size() {
      var w = Math.round(wrap.clientWidth);
      if (!w) return false;
      var ht = typeof o.height === 'function' ? o.height(w) : o.aspect ? w * o.aspect : (o.height || 300);
      if (o.minH) ht = Math.max(o.minH, ht);
      if (o.maxH) ht = Math.min(o.maxH, ht);
      ht = Math.round(ht);
      var dpr = Math.min(window.devicePixelRatio || 1, o.dprCap || 2);
      if (w === s.w && ht === s.h && dpr === s.dpr) return false;
      s.w = w; s.h = ht; s.dpr = dpr;
      s.sc = M.clamp(w / (o.ref || 560), 0.82, 1.5);
      [cv, ov].forEach(function (c) {
        if (!c) return;
        c.style.height = ht + 'px';
        c.width = Math.round(w * dpr);
        c.height = Math.round(ht * dpr);
        c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
      });
      if (s.onResize) s.onResize(s);
      return true;
    }
    var raf = 0;
    s.request = function () {
      // First paint is synchronous: background tabs may never run rAF, and
      // readouts are filled by draw().
      if (!s.drawn) {
        if (!s.w) size();
        if (s.w && s.draw) { s.drawn = true; s.draw(s.ctx, s); return; }
      }
      if (raf) return;
      raf = requestAnimationFrame(function () {
        raf = 0;
        if (!s.w) size();
        if (s.w && s.draw) s.draw(s.ctx, s);
      });
    };
    var themePaint = function () {
      if (!wrap.isConnected) { document.removeEventListener('ml4hl-theme', themePaint); return; }
      if (s.draw) s.draw(s.ctx, s);
    };
    document.addEventListener('ml4hl-theme', themePaint);
    s.resize = function () { if (size()) { if (s.draw) s.draw(s.ctx, s); M.colabResize(); } };
    if (window.ResizeObserver) new ResizeObserver(function () { s.resize(); }).observe(wrap);
    else window.addEventListener('resize', s.resize);
    size();
    return s;
  };

  M.font = function (s, px, weight, family) {
    return (weight || 400) + ' ' + Math.round(px * (s ? s.sc : 1) * 10) / 10 + 'px ' +
      (family === 'sans' ? '"DM Sans", system-ui, sans-serif' : '"DM Mono", ui-monospace, Menlo, monospace');
  };

  M.clear = function (ctx, s, color) {
    ctx.clearRect(0, 0, s.w, s.h);
    ctx.fillStyle = color || M.C.bg;
    ctx.fillRect(0, 0, s.w, s.h);
  };

  /* Data-to-pixel mapping inside a padded plot rectangle. */
  M.view = function (s, o) {
    var base = { x: o.x.slice(), y: o.y.slice() };
    var v = { pad: Object.assign({ l: 52, r: 14, t: 12, b: 38 }, o.pad || {}), x0: 0, x1: 1, y0: 0, y1: 1 };
    v.setDomain = function (x, y) { base = { x: x.slice(), y: y.slice() }; v.update(); };
    v.update = function (rect) {
      var p = v.pad, sc = s.sc;
      if (rect) { v.L = rect.x; v.R = rect.x + rect.w; v.T = rect.y; v.B = rect.y + rect.h; }
      else { v.L = p.l * sc; v.R = s.w - p.r * sc; v.T = p.t * sc; v.B = s.h - p.b * sc; }
      v.x0 = base.x[0]; v.x1 = base.x[1]; v.y0 = base.y[0]; v.y1 = base.y[1];
      if (o.equal) {
        var ux = (v.x1 - v.x0) / (v.R - v.L), uy = (v.y1 - v.y0) / (v.B - v.T);
        if (ux > uy) { var cy = (v.y0 + v.y1) / 2, hy = ux * (v.B - v.T) / 2; v.y0 = cy - hy; v.y1 = cy + hy; }
        else { var cx = (v.x0 + v.x1) / 2, hx = uy * (v.R - v.L) / 2; v.x0 = cx - hx; v.x1 = cx + hx; }
      }
      return v;
    };
    v.px = function (x) { return v.L + (x - v.x0) / (v.x1 - v.x0) * (v.R - v.L); };
    v.py = function (y) { return v.B - (y - v.y0) / (v.y1 - v.y0) * (v.B - v.T); };
    v.ix = function (px) { return v.x0 + (px - v.L) / (v.R - v.L) * (v.x1 - v.x0); };
    v.iy = function (py) { return v.y0 + (v.B - py) / (v.B - v.T) * (v.y1 - v.y0); };
    v.inside = function (px, py) { return px >= v.L && px <= v.R && py >= v.T && py <= v.B; };
    v.clip = function (ctx) { ctx.beginPath(); ctx.rect(v.L, v.T, v.R - v.L, v.B - v.T); ctx.clip(); };
    return v;
  };

  M.ticks = function (a, b, n) {
    var span = b - a;
    if (!(span > 0)) return [a];
    var step = Math.pow(10, Math.floor(Math.log10(span / (n || 5))));
    var err = (n || 5) / span * step;
    if (err <= 0.15) step *= 10; else if (err <= 0.35) step *= 5; else if (err <= 0.75) step *= 2;
    var out = [], t = Math.ceil(a / step) * step;
    for (; t <= b + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    return out;
  };
  function tickLabel(t) {
    var a = Math.abs(t);
    if (a >= 10000) return (t / 1000).toFixed(0) + 'k';
    if (a >= 100 || Number.isInteger(t)) return M.f(t, 0);
    if (a >= 1) return M.f(t, 1).replace(/\.0$/, '');
    return M.f(t, 2).replace(/0$/, '');
  }

  M.axes = function (ctx, v, s, o) {
    o = o || {};
    var xt = o.xticks || M.ticks(v.x0, v.x1, o.nx || Math.max(3, Math.round((v.R - v.L) / (80 * s.sc))));
    var yt = o.yticks || M.ticks(v.y0, v.y1, o.ny || Math.max(3, Math.round((v.B - v.T) / (60 * s.sc))));
    ctx.save();
    ctx.lineWidth = 1;
    if (o.grid !== false) {
      ctx.strokeStyle = M.C.grid;
      ctx.beginPath();
      if (o.grid !== 'y') xt.forEach(function (t) { var x = Math.round(v.px(t)) + 0.5; ctx.moveTo(x, v.T); ctx.lineTo(x, v.B); });
      if (o.grid !== 'x') yt.forEach(function (t) { var y = Math.round(v.py(t)) + 0.5; ctx.moveTo(v.L, y); ctx.lineTo(v.R, y); });
      ctx.stroke();
    }
    ctx.strokeStyle = M.C.faint;
    ctx.beginPath();
    ctx.moveTo(v.L + 0.5, v.T); ctx.lineTo(v.L + 0.5, v.B + 0.5); ctx.lineTo(v.R, v.B + 0.5);
    ctx.stroke();
    ctx.fillStyle = M.C.muted;
    ctx.font = M.font(s, 9.5);
    if (o.xlabels !== false) {
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      xt.forEach(function (t) { ctx.fillText(o.xfmt ? o.xfmt(t) : tickLabel(t), v.px(t), v.B + 5 * s.sc); });
    }
    if (o.ylabels !== false) {
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      yt.forEach(function (t) { ctx.fillText(o.yfmt ? o.yfmt(t) : tickLabel(t), v.L - 6 * s.sc, v.py(t)); });
    }
    ctx.font = M.font(s, 9, 500);
    ctx.fillStyle = M.C.muted;
    if (o.xlabel) {
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(o.xlabel.toUpperCase(), (v.L + v.R) / 2, v.B + (o.xlabels !== false ? 19 : 7) * s.sc);
    }
    if (o.ylabel) {
      var yw = 0;
      if (o.ylabels !== false) {
        ctx.font = M.font(s, 9.5);
        yt.forEach(function (t) { yw = Math.max(yw, ctx.measureText(o.yfmt ? o.yfmt(t) : tickLabel(t)).width); });
        ctx.font = M.font(s, 9, 500);
      }
      ctx.save();
      ctx.translate(Math.max(2 * s.sc, v.L - (yw ? yw + 10 * s.sc : 6 * s.sc)), (v.T + v.B) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(o.ylabel.toUpperCase(), 0, 0);
      ctx.restore();
    }
    ctx.restore();
  };

  M.dot = function (ctx, x, y, r, fill, stroke, lw) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
  };
  M.star = function (ctx, x, y, r, fill, stroke) {
    ctx.beginPath();
    for (var i = 0; i < 10; i++) {
      var a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
      ctx[i ? 'lineTo' : 'moveTo'](x + rr * Math.cos(a), y + rr * Math.sin(a));
    }
    ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.stroke(); }
  };
  M.pill = function (ctx, s, text, x, y, o) {
    o = o || {};
    ctx.save();
    ctx.font = M.font(s, o.size || 10, o.weight || 500);
    var w = ctx.measureText(text).width + 10 * s.sc, ht = 17 * s.sc;
    var bx = o.align === 'right' ? x - w : o.align === 'center' ? x - w / 2 : x;
    ctx.fillStyle = o.bg || M.rgba(M.C.surface, 0.94);
    ctx.strokeStyle = o.border || M.C.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(bx, y - ht / 2, w, ht, 3 * s.sc); else ctx.rect(bx, y - ht / 2, w, ht);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = o.color || M.C.ink;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + 5 * s.sc, y + 0.5);
    ctx.restore();
  };
  M.arrow = function (ctx, x0, y0, x1, y1, color, lw, head) {
    var a = Math.atan2(y1 - y0, x1 - x0), hl = head || 7;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw || 1.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * hl * 0.6, y1 - Math.sin(a) * hl * 0.6); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - hl * Math.cos(a - 0.4), y1 - hl * Math.sin(a - 0.4));
    ctx.lineTo(x1 - hl * Math.cos(a + 0.4), y1 - hl * Math.sin(a + 0.4));
    ctx.closePath(); ctx.fill();
  };

  /* Paint a function of (x, y) over the plot rectangle at low resolution and
     upscale it smoothly. fn returns [r, g, b, a(0-255)]. */
  M.field = function (ctx, v, nx, ny, fn, smooth) {
    var off = document.createElement('canvas');
    off.width = nx; off.height = ny;
    var octx = off.getContext('2d'), img = octx.createImageData(nx, ny), d = img.data;
    for (var j = 0; j < ny; j++) {
      var y = v.y1 - (j + 0.5) / ny * (v.y1 - v.y0);
      for (var i = 0; i < nx; i++) {
        var x = v.x0 + (i + 0.5) / nx * (v.x1 - v.x0);
        var c = fn(x, y, i, j), k = (j * nx + i) * 4;
        d[k] = c[0]; d[k + 1] = c[1]; d[k + 2] = c[2]; d[k + 3] = c[3] == null ? 255 : c[3];
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = smooth !== false;
    ctx.drawImage(off, v.L, v.T, v.R - v.L, v.B - v.T);
    ctx.restore();
  };

  /* ─── Pointer input (mouse, pen, touch) ────────────────────────── */

  M.pointer = function (el, hd) {
    if (hd.touch !== false) el.style.touchAction = 'none';
    var active = false;
    function pos(e) { var r = el.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
    el.addEventListener('pointerdown', function (e) {
      if (e.button != null && e.button > 0 && !hd.anyButton) return;
      var res = hd.down ? hd.down(pos(e), e) : true;
      if (res !== false) { active = true; try { el.setPointerCapture(e.pointerId); } catch (err) { /* noop */ } }
    });
    el.addEventListener('pointermove', function (e) {
      var p = pos(e);
      if (active) { if (hd.drag) hd.drag(p, e); }
      else if (hd.hover) hd.hover(p, e);
    });
    function end(e) { if (!active) return; active = false; if (hd.up) hd.up(pos(e), e); }
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', function (e) { if (!active && hd.leave) hd.leave(e); });
    if (hd.wheel) el.addEventListener('wheel', function (e) { e.preventDefault(); hd.wheel(pos(e), e); }, { passive: false });
    if (hd.key) {
      el.tabIndex = 0;
      el.addEventListener('keydown', function (e) { if (hd.key(e) === true) e.preventDefault(); });
    }
  };

  /* ─── Time: animation loops, transports, visibility ────────────── */

  M.reducedMotion = function () {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  };

  M.onVisible = function (el, cb) {
    if (!window.IntersectionObserver) { cb(true); return; }
    new IntersectionObserver(function (es) { es.forEach(function (e) { cb(e.isIntersecting); }); }, { threshold: 0.05 }).observe(el);
  };

  /* A frame loop: step(dt) is called every animation frame until it returns false. */
  M.loop = function (step) {
    var on = false, last = 0;
    function f(t) {
      if (!on) return;
      var dt = last ? Math.min(0.05, (t - last) / 1000) : 1 / 60;
      last = t;
      if (step(dt) === false) { on = false; return; }
      requestAnimationFrame(f);
    }
    return {
      start: function () { if (on) return; on = true; last = 0; requestAnimationFrame(f); },
      stop: function () { on = false; },
      get on() { return on; }
    };
  };

  /* Play / pause button plus a scrubbable step slider. */
  M.transport = function (parent, o) {
    var row = parent.appendChild(h('div', { class: 'wrow transport' }));
    var btn = h('button', { type: 'button', class: 'wbtn wplay', 'aria-label': 'Play or pause', 'data-role': 'play' }, ['play']);
    var input = h('input', { type: 'range', min: 0, max: 0, step: 1, value: 0, 'aria-label': o.aria || 'Step', class: 'wscrub' });
    var out = h('span', { class: 'readout wval tval' });
    row.appendChild(btn); row.appendChild(input); row.appendChild(out);
    var n = 1, i = 0, playing = false, last = 0, visible = true, wantPlay = false;
    var fps = o.fps || 30;
    function show(k, src) {
      i = M.clamp(Math.round(k), 0, n - 1);
      input.value = i;
      out.innerHTML = o.fmt ? o.fmt(i, n) : 'step ' + i + ' / ' + (n - 1);
      if (o.onFrame) o.onFrame(i, src || 'set');
    }
    function tick(t) {
      if (!playing) return;
      if (!visible) { last = 0; requestAnimationFrame(tick); return; }
      if (!last) last = t;
      var due = Math.floor((t - last) * fps / 1000);
      if (due >= 1) {
        last += due * 1000 / fps;
        var stride = typeof o.stride === 'function' ? o.stride(i, n) : (o.stride || 1);
        var nxt = i + Math.max(1, Math.min(due, 4)) * stride;
        if (nxt >= n - 1) {
          show(n - 1, 'play');
          if (o.loop) { last = t + (o.hold || 900); setTimeout(function () { if (playing) show(0, 'play'); }, o.hold || 900); }
          else { pause(); return; }
        } else show(nxt, 'play');
      }
      requestAnimationFrame(tick);
    }
    function play() {
      if (n <= 1) return;
      if (i >= n - 1) show(0, 'play');
      playing = true; wantPlay = true; btn.textContent = 'pause'; btn.classList.add('on');
      last = 0; requestAnimationFrame(tick);
      if (o.onPlay) o.onPlay(true);
    }
    function pause() {
      playing = false; wantPlay = false; btn.textContent = 'play'; btn.classList.remove('on');
      if (o.onPlay) o.onPlay(false);
    }
    btn.addEventListener('click', function () { if (playing) pause(); else play(); });
    input.addEventListener('input', function () { pause(); show(+input.value, 'scrub'); });
    var api = {
      row: row, play: play, pause: pause, show: show,
      setN: function (m, keep) { n = Math.max(1, m | 0); input.max = n - 1; show(keep ? Math.min(i, n - 1) : 0, 'reset'); },
      get i() { return i; }, get n() { return n; }, get playing() { return playing; }
    };
    api.setN(o.n || 1);
    M.onVisible(row, function (vis) {
      visible = vis;
      if (vis && o.autoplay && !wantPlay && !api.__started && !M.reducedMotion()) { api.__started = true; play(); }
    });
    return api;
  };

  /* ─── Host integration ─────────────────────────────────────────── */

  var resizeTimer = 0;
  M.colabResize = function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      try {
        var out = window.google && window.google.colab && window.google.colab.output;
        if (out && out.resizeIframeToContent) out.resizeIframeToContent();
      } catch (err) { /* not in Colab */ }
    }, 80);
  };

  /* Split container: two panels side by side, stacked on narrow screens. */
  M.split = function (parent, ratio, cls) {
    var el = parent.appendChild(h('div', { class: 'wsplit' + (cls ? ' ' + cls : '') }));
    var a = el.appendChild(h('div', { class: 'wpanel', style: { flex: (ratio || 1) + ' 1 0' } }));
    var b = el.appendChild(h('div', { class: 'wpanel', style: { flex: '1 1 0' } }));
    return { el: el, a: a, b: b };
  };
  M.mini = function (parent, text) { return parent.appendChild(h('div', { class: 'mini-title', html: text })); };
})();
