(function () {
  var root = document.documentElement;
  var button = document.getElementById('theme-toggle');
  var stored = localStorage.getItem('theme');
  var preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  var theme = stored || preferred;

  root.dataset.theme = theme;

  var seasonalHero = document.querySelector('[data-seasonal-hero]');
  if (seasonalHero) {
    var weightedSeasons = [
      'spring', 'spring', 'spring',
      'summer', 'summer', 'summer',
      'autumn', 'autumn', 'autumn',
      'winter'
    ];
    var season = weightedSeasons[Math.floor(Math.random() * weightedSeasons.length)];
    var seasonKey = season.charAt(0).toUpperCase() + season.slice(1);
    var seasonalTitle = document.getElementById('home-title');
    var seasonalArt = document.getElementById('home-season-art');

    seasonalHero.dataset.season = season;
    root.dataset.homeSeason = season;

    if (seasonalTitle) {
      seasonalTitle.textContent = seasonalHero.dataset['title' + seasonKey];
    }

    if (seasonalArt) {
      seasonalArt.addEventListener('load', function () {
        seasonalArt.hidden = false;
      }, { once: true });
      seasonalArt.width = Number(seasonalHero.dataset['width' + seasonKey]);
      seasonalArt.height = Number(seasonalHero.dataset['height' + seasonKey]);
      seasonalArt.src = seasonalHero.dataset['image' + seasonKey];
      seasonalArt.alt = seasonalHero.dataset['alt' + seasonKey];
    }
  }

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
