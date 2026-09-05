(function () {
  var panel = document.getElementById('comments');
  if (!panel) return;
  var load = document.getElementById('load-comments');
  var more = document.getElementById('more-comments');
  var status = document.getElementById('comments-status');
  var list = document.getElementById('comment-list');
  var discussion = document.getElementById('discussion-link');
  var repository = 'Camellia86/Camellia86.github.io';
  var issue;
  var page = 1;
  var busy = false;
  var seen = new Set();
  load.hidden = false;

  var request = async function (url) {
    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 15000);
    try {
      var response = await fetch(url, { signal: controller.signal, credentials: 'omit', referrerPolicy: 'no-referrer' });
      if (!response.ok) throw new Error(response.status === 403 || response.status === 429 ? 'rate-limit' : 'unavailable');
      return await response.json();
    } finally { clearTimeout(timeout); }
  };

  var getThread = async function () {
    var index = await request(panel.dataset.commentIndex);
    var post = index.posts.find(function (entry) { return entry.path === panel.dataset.commentPath; });
    if (!post || !Number.isInteger(post.issue) || post.issue < 1) throw new Error('not-published');
    issue = post.issue;
    discussion.href = 'https://github.com/' + repository + '/issues/' + issue + '#new_comment_field';
  };

  var render = function (comment) {
    if (seen.has(comment.id)) return;
    seen.add(comment.id);
    var card = document.createElement('article');
    card.className = 'comment-card';
    var header = document.createElement('header');
    var author = document.createElement('strong');
    author.textContent = comment.user ? comment.user.login : 'Deleted account';
    var date = document.createElement('a');
    date.href = 'https://github.com/' + repository + '/issues/' + issue + '#issuecomment-' + Number(comment.id);
    date.target = '_blank';
    date.rel = 'noopener noreferrer';
    var time = document.createElement('time');
    time.dateTime = comment.created_at;
    time.textContent = new Date(comment.created_at).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    date.append(time);
    header.append(author, date);
    var body = document.createElement('p');
    body.className = 'comment-body';
    body.textContent = comment.body || '(No text content)';
    card.append(header, body);
    list.append(card);
  };

  var loadPage = async function (reset) {
    if (busy) return;
    busy = true;
    load.disabled = true;
    more.disabled = true;
    panel.setAttribute('aria-busy', 'true');
    status.textContent = 'Loading the conversation…';
    if (reset) { page = 1; seen.clear(); list.replaceChildren(); more.hidden = true; }
    try {
      if (!issue) await getThread();
      var comments = await request('https://api.github.com/repos/' + repository + '/issues/' + issue + '/comments?per_page=30&page=' + page);
      if (!Array.isArray(comments)) throw new Error('unavailable');
      comments.forEach(render);
      page++;
      more.hidden = comments.length < 30;
      status.textContent = seen.size ? seen.size + (seen.size === 1 ? ' comment shown.' : ' comments shown.') : 'No comments yet. Start the conversation on GitHub.';
      load.textContent = 'Refresh comments';
    } catch (error) {
      status.textContent = error.message === 'not-published'
        ? 'Discussion links are prepared during deployment. You can also check the conversation on GitHub.'
        : error.message === 'rate-limit'
          ? 'GitHub’s public request limit has been reached. Please try later or read the discussion on GitHub.'
          : 'Could not load comments. Try again, or open the discussion on GitHub.';
      load.textContent = 'Retry comments';
    } finally {
      busy = false;
      load.disabled = false;
      more.disabled = false;
      panel.removeAttribute('aria-busy');
    }
  };

  getThread().catch(function () {});
  load.addEventListener('click', function () { loadPage(true); });
  more.addEventListener('click', function () { loadPage(false); });
})();
