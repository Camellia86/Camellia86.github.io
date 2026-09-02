(function () {
  var root = document.documentElement;
  var button = document.getElementById('theme-toggle');
  var stored = localStorage.getItem('theme');
  var preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  var theme = stored || preferred;

  root.dataset.theme = theme;

  if (button) {
    button.addEventListener('click', function () {
      var next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('theme', next);
    });
  }

  var progress = document.getElementById('reading-progress');
  var article = document.querySelector('.post');
  if (progress && article) {
    var updateProgress = function () {
      var max = document.documentElement.scrollHeight - window.innerHeight;
      var value = max > 0 ? Math.min(window.scrollY / max, 1) : 0;
      progress.style.transform = 'scaleX(' + value + ')';
    };
    updateProgress();
    window.addEventListener('scroll', updateProgress, { passive: true });
    window.addEventListener('resize', updateProgress);
  }
})();

