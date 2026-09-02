# Camellia86

Source for [camellia86.github.io](https://camellia86.github.io/), a reading-focused Hugo blog for research notes on generative modeling and stochastic dynamics.

## Local preview

1. Install Hugo Extended.
2. Run `hugo server -D`.
3. Open `http://localhost:1313/`.

## Publish a new article

Create `content/blog/<slug>/index.md` with TOML front matter:

```toml
+++
title = 'Article title'
date = 2026-09-02

[params]
subtitle = 'A short subtitle'
math = true
+++
```

Push to `main`. GitHub Actions builds the site and publishes it through GitHub Pages.

