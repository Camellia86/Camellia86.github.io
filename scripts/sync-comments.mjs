import { readFile, writeFile } from 'node:fs/promises';

const manifestPath = 'public/comments-index.json';
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const repository = process.env.GITHUB_REPOSITORY;
const token = process.env.GITHUB_TOKEN;
if (!token || repository !== manifest.repository) throw new Error('A repository-scoped GitHub Actions token is required.');

async function github(path, body) {
  const response = await fetch(`https://api.github.com/repos/${repository}/${path}`, {
    method: body ? 'POST' : 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'Content-Type': 'application/json',
      'X-GitHub-Api-Version': '2022-11-28'
    },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(30000)
  });
  if (!response.ok) throw new Error(`GitHub ${response.status} while syncing comment threads. Check Issues and workflow permissions.`);
  return response.json();
}

const threads = [];
for (let page = 1; ; page++) {
  const issues = await github(`issues?state=all&per_page=100&page=${page}`);
  threads.push(...issues.filter(issue => !issue.pull_request));
  if (issues.length < 100) break;
}

for (const post of manifest.posts) {
  const marker = `<!-- blog-comment-thread:${post.path} -->`;
  let thread = threads.find(issue => issue.body?.includes(marker));
  if (!thread) {
    thread = await github('issues', {
      title: `Discussion: ${post.title}`.slice(0, 256),
      body: `${marker}\n\n## ${post.title}\n\nRead the article: ${post.url}\n\nQuestions, corrections, and different perspectives are welcome. Comments here are publicly displayed on the blog. Please keep the conversation thoughtful and on topic.`
    });
    threads.push(thread);
  }
  post.issue = thread.number;
  console.log(`Comment thread ready: ${post.path} → #${thread.number}`);
}

await writeFile(manifestPath, JSON.stringify(manifest));
