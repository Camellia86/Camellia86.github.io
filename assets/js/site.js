(function () {
  var root = document.documentElement;
  var button = document.getElementById('theme-toggle');
  var stored;
  try { stored = localStorage.getItem('theme'); } catch (error) {}
  var preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  var theme = stored === 'dark' || stored === 'light' ? stored : preferred;

  root.dataset.theme = theme;

  var seasonalHero = document.querySelector('[data-seasonal-hero]');
  if (seasonalHero) {
    var seasons = ['winter', 'spring', 'summer', 'autumn'];
    var season = seasons[Math.floor(((new Date().getMonth() + 1) % 12) / 3)];
    try {
      var savedSeason = localStorage.getItem('season');
      if (seasons.includes(savedSeason)) season = savedSeason;
    } catch (error) {}
    var seasonalTitle = document.getElementById('home-title');
    var seasonalArt = document.getElementById('home-season-art');
    var choices = seasonalHero.querySelectorAll('[data-season-choice]');
    var setSeason = function (next) {
      var seasonKey = next.charAt(0).toUpperCase() + next.slice(1);
      seasonalHero.dataset.season = next;
      root.dataset.homeSeason = next;
      seasonalTitle.textContent = seasonalHero.dataset['title' + seasonKey];
      seasonalArt.width = Number(seasonalHero.dataset['width' + seasonKey]);
      seasonalArt.height = Number(seasonalHero.dataset['height' + seasonKey]);
      seasonalArt.src = seasonalHero.dataset['image' + seasonKey];
      seasonalArt.alt = seasonalHero.dataset['alt' + seasonKey];
      choices.forEach(function (choice) {
        choice.setAttribute('aria-pressed', String(choice.dataset.seasonChoice === next));
      });
    };
    setSeason(season);
    seasonalHero.querySelector('.season-picker').hidden = false;
    choices.forEach(function (choice) {
      choice.addEventListener('click', function () {
        setSeason(choice.dataset.seasonChoice);
        try { localStorage.setItem('season', choice.dataset.seasonChoice); } catch (error) {}
      });
    });
  }

  if (button) {
    button.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    button.addEventListener('click', function () {
      var next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      button.setAttribute('aria-label', next === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
      try { localStorage.setItem('theme', next); } catch (error) {}
    });
  }

  var progress = document.getElementById('reading-progress');
  var article = document.querySelector('.post .prose');
  if (progress && article) {
    var updateProgress = function () {
      var bounds = article.getBoundingClientRect();
      var max = article.scrollHeight - window.innerHeight + 160;
      var value = max > 0 ? Math.max(0, Math.min((160 - bounds.top) / max, 1)) : (bounds.top < 160 ? 1 : 0);
      progress.style.transform = 'scaleX(' + value + ')';
    };
    updateProgress();
    window.addEventListener('scroll', updateProgress, { passive: true });
    window.addEventListener('resize', updateProgress);
    if ('ResizeObserver' in window) new ResizeObserver(updateProgress).observe(article);
  }
})();
