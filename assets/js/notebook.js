(function () {
  var root = document.documentElement;
  var toolbar = document.querySelector('.reader-toolbar');
  if (toolbar) {
    toolbar.hidden = false;
    var size = 100;
    try { size = Number(localStorage.getItem('reading-size')) || 100; } catch (error) {}
    var sizeButtons = toolbar.querySelectorAll('[data-font-step]');
    var setSize = function (next) {
      size = Math.max(80, Math.min(160, next));
      root.style.setProperty('--reading-scale', size / 100);
      document.getElementById('font-size-value').textContent = size + '%';
      sizeButtons.forEach(function (control) {
        control.disabled = Number(control.dataset.fontStep) < 0 ? size <= 80 : size >= 160;
      });
      try { localStorage.setItem('reading-size', String(size)); } catch (error) {}
    };
    setSize(size);
    sizeButtons.forEach(function (control) {
      control.addEventListener('click', function () { setSize(size + Number(control.dataset.fontStep)); });
    });
    document.getElementById('font-reset').addEventListener('click', function () { setSize(100); });
    var focusToggle = document.getElementById('focus-toggle');
    focusToggle.addEventListener('click', function () {
      var enabled = root.classList.toggle('focus-reading');
      focusToggle.setAttribute('aria-pressed', String(enabled));
      focusToggle.textContent = enabled ? 'Exit focus' : 'Focus mode';
    });
  }

  var search = document.getElementById('writing-search');
  if (search) {
    document.querySelector('.writing-search').hidden = false;
    var cards = Array.from(document.querySelectorAll('.article-card'));
    var clear = document.getElementById('clear-search');
    var normalize = function (text) {
      return text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
    };
    var filter = function () {
      var terms = normalize(search.value.trim()).split(/\s+/).filter(Boolean);
      var count = 0;
      cards.forEach(function (card) {
        var matches = terms.every(function (term) { return normalize(card.dataset.searchText).includes(term); });
        card.hidden = !matches;
        if (matches) count++;
      });
      clear.hidden = !search.value;
      document.getElementById('search-status').textContent = terms.length
        ? (count ? count + (count === 1 ? ' note found' : ' notes found') : 'No notes found. Try a different topic or clear your search.')
        : cards.length + (cards.length === 1 ? ' note in the notebook' : ' notes in the notebook');
    };
    search.addEventListener('input', filter);
    clear.addEventListener('click', function () { search.value = ''; filter(); search.focus(); });
    filter();
  }

  var headings = Array.from(document.querySelectorAll('.post .prose h2[id], .post .prose h3[id]'));
  var tocLinks = document.querySelectorAll('.post-toc a');
  if (headings.length && tocLinks.length) {
    var scheduled = false;
    var markSection = function () {
      scheduled = false;
      var current = headings[0];
      headings.forEach(function (heading) { if (heading.getBoundingClientRect().top <= 190) current = heading; });
      tocLinks.forEach(function (link) {
        var active = decodeURIComponent(link.hash.slice(1)) === current.id;
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    };
    window.addEventListener('scroll', function () {
      if (!scheduled) { scheduled = true; requestAnimationFrame(markSection); }
    }, { passive: true });
    markSection();
  }
})();
