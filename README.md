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

Write inline mathematics as `$x_t$`. Protect display mathematics from
Markdown parsing with the `math` shortcode:

```text
{{< math >}}
\begin{aligned}
x &= y + z \\
u &= v
\end{aligned}
{{< /math >}}
```

Push to `main`. GitHub Actions builds the site and publishes it through GitHub Pages.

## Reading experience

- Article toolbar: text size from 80% to 160%, reset, and distraction-free focus mode.
- Text size, color theme, and the homepage season are remembered on the current device.
- Homepage and archive search runs locally over titles, topics, and summaries (accent-insensitive).
- Seasonal artwork, topic labels, active table of contents, and article-only reading progress.
- No external fonts or analytics. Controls support keyboard navigation and reduced motion.

## Comments

Each published blog article has a GitHub Issues discussion. Readers load comments
on demand, then sign in **on GitHub** to post; no credentials are stored in the blog.
The blog displays comment text safely as plain text; full Markdown and attachments
remain available on GitHub. There is no inline login or anonymous posting.

During deployment, `scripts/sync-comments.mjs` reads Hugo's `public/comments-index.json`,
reuses issues using a stable article-path marker, and creates missing threads with
the job-scoped `GITHUB_TOKEN` (`issues: write`). It writes issue numbers only into
the deployment artifact. No personal access token or GitHub App installation is needed.
Keep repository Issues enabled and do not remove the marker in a thread's first post.
Changing an article's URL creates a new thread; migrate its marker before publishing
if the old discussion should follow the new URL. Moderate comments in repository Issues.

Local `hugo server` previews show the comments UI but do not create remote issues.
Production deployment prepares the direct discussion links. Network errors and GitHub
public API rate limits keep a direct GitHub fallback available. Comments paginate in
batches of 30; refresh reloads the conversation. No GitHub request is made before
the reader chooses to load comments or follows a GitHub link.
