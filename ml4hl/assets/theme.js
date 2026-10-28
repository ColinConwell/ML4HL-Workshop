(function () {
  'use strict';
  if (window.ML4HLTheme) return;
  var config = window.ML4HL_THEME_CONFIG, media = matchMedia('(prefers-color-scheme: dark)');
  var saved = {};
  try { saved = JSON.parse(localStorage.getItem('ml4hl-appearance') || '{}'); } catch (_) {}
  var state = {theme: ['light','dark','auto'].includes(saved.theme) ? saved.theme : 'auto', plots: saved.plots === 'original' ? 'original' : 'follow'};
  function apply() {
    var mode = state.theme === 'auto' ? (media.matches ? 'dark' : 'light') : state.theme;
    var root = document.documentElement, c = config[mode];
    root.dataset.ml4hlTheme = mode; root.dataset.ml4hlPlots = state.plots;
    root.style.colorScheme = mode;
    Object.keys(c).forEach(function (key) { root.style.setProperty('--ml-' + key, c[key]); });
    if (document.body) {
      document.body.classList.toggle('quarto-dark', mode === 'dark');
      document.body.classList.toggle('quarto-light', mode === 'light');
    }
    // Let Quarto's own theme switch its syntax highlighting and Bootstrap rules.
    document.querySelectorAll('link.quarto-color-scheme').forEach(function (link) {
      link.rel = !link.classList.contains('quarto-color-alternate') || mode === 'dark' ? 'stylesheet' : 'prefetch';
    });
    if (window.ML4HL) {
      Object.assign(ML4HL.C, c);
      ML4HL.cat.splice(0, ML4HL.cat.length, ...config.categorical.map(function(k){return c[k];}));
    }
    document.querySelectorAll('[data-ml-theme-control]').forEach(function(el){el.value=state.theme;});
    document.querySelectorAll('[data-ml-plot-control]').forEach(function(el){el.value=state.plots;});
    document.dispatchEvent(new CustomEvent('ml4hl-theme', {detail:{mode:mode}}));
    try { localStorage.setItem('ml4hl-appearance', JSON.stringify(state)); localStorage.setItem('quarto-color-scheme', mode === 'dark' ? 'alternate' : 'default'); } catch (_) {}
  }
  function controls(parent) {
    if (document.querySelector('.ml4hl-appearance')) return;
    var bar = document.createElement('div'); bar.className = 'ml4hl-appearance';
    bar.innerHTML = '<label>Page Theme <select aria-label="Page Theme" data-ml-theme-control><option value="auto">Auto</option><option value="light">Light</option><option value="dark">Dark</option></select></label><label>Plot Style <select aria-label="Plot Style" data-ml-plot-control><option value="follow">Follow Page Theme</option><option value="original">Keep Original Plot Style</option></select></label>';
    parent.prepend(bar);
    bar.querySelector('[data-ml-theme-control]').addEventListener('change',function(e){state.theme=e.target.value;apply();});
    bar.querySelector('[data-ml-plot-control]').addEventListener('change',function(e){state.plots=e.target.value;apply();});
  }
  function ready() {
    var parent = document.querySelector('.g-main') || document.querySelector('main.content') || document.body;
    controls(parent);
    var title = document.querySelector('.quarto-title');
    if (title && window.ML4HL_LOGO && !title.querySelector('.ml4hl-logo')) {
      var img = document.createElement('img'); img.className='ml4hl-logo'; img.alt='VISNA Lab'; img.src=window.ML4HL_LOGO; title.append(img);
    }
    apply();
    setTimeout(apply, 0); // Quarto also initializes at DOMContentLoaded.
  }
  window.ML4HLTheme = {apply:apply, state:state};
  media.addEventListener('change',function(){if(state.theme==='auto') apply();});
  apply();
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded',ready); else ready();
})();
