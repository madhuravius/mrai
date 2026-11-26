// Loaded in <head> (not deferred) so a saved theme applies before first paint;
// everything touching <body> waits for DOMContentLoaded. index.js and
// highlight.js are deferred, so both have run by then.
(function () {
  'use strict';

  var root = document.documentElement;
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* file:// may refuse */ } },
  };

  // ---- theme -------------------------------------------------------------
  function systemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function currentTheme() { return root.dataset.theme || systemTheme(); }
  function applyTheme(theme) {
    root.dataset.theme = theme;
    document.getElementById('hl-light').media = theme === 'light' ? 'all' : 'not all';
    document.getElementById('hl-dark').media = theme === 'dark' ? 'all' : 'not all';
  }
  var savedTheme = store.get('mrai-theme');
  if (savedTheme) applyTheme(savedTheme);

  // nav state must also be set before paint to avoid a layout jump
  var savedNav = store.get('mrai-nav');
  if (savedNav) root.dataset.nav = savedNav;

  // ---- helpers -------------------------------------------------------------
  var UNITS = [[31536000, 'year'], [2592000, 'month'], [604800, 'week'], [86400, 'day'],
    [3600, 'hour'], [60, 'minute']];

  function ago(iso) {
    var secs = (Date.now() - Date.parse(iso)) / 1000;
    if (isNaN(secs)) return '';
    for (var i = 0; i < UNITS.length; i++) {
      var n = Math.floor(secs / UNITS[i][0]);
      if (n >= 1) return n + ' ' + UNITS[i][1] + (n > 1 ? 's' : '') + ' ago';
    }
    return 'just now';
  }

  // earliest occurrence of any term at or after `from`: [index, length], index -1 if none
  function firstHit(low, terms, from) {
    var best = -1, len = 0;
    terms.forEach(function (t) {
      var i = low.indexOf(t, from);
      if (i >= 0 && (best < 0 || i < best)) { best = i; len = t.length; }
    });
    return [best, len];
  }

  function describe(text, terms) {
    var flat = text.replace(/\*\*Q:\*\*/, '').replace(/[*_`#>]/g, '').replace(/\s+/g, ' ').trim();
    var at = Math.max(0, firstHit(flat.toLowerCase(), terms, 0)[0]);
    var s = Math.max(0, at - 30);
    return (s ? '…' : '') + flat.slice(s, s + 100) + (s + 100 < flat.length ? '…' : '');
  }

  function highlightTerms(container, terms) {
    var walk = document.createTreeWalker(container, NodeFilter.SHOW_TEXT), nodes = [];
    while (walk.nextNode()) nodes.push(walk.currentNode);
    nodes.forEach(function (node) {
      var text = node.nodeValue, low = text.toLowerCase(), pos = 0, hit;
      var frag = document.createDocumentFragment();
      while ((hit = firstHit(low, terms, pos))[0] >= 0) {
        frag.append(text.slice(pos, hit[0]));
        var m = document.createElement('mark');
        m.textContent = text.slice(hit[0], hit[0] + hit[1]);
        frag.append(m);
        pos = hit[0] + hit[1];
      }
      if (!pos) return;
      frag.append(text.slice(pos));
      node.replaceWith(frag);
    });
  }

  function label(e) {
    return e.title + (e.heading && e.heading !== e.title ? ' › ' + e.heading : '');
  }

  function metaText(e) {
    return [ago(e.ts), e.cwd, e.file.split('/').pop()].filter(Boolean).join(' · ');
  }

  function fillMeta(el, e) {
    el.textContent = metaText(e);
    el.title = new Date(e.ts).toLocaleString() + '\n' + e.file;
  }

  // every page sits in the site folder itself (no subfolders: Safari is strict
  // about local pages loading files from other folders)
  function pageFile(e) { return e.ws + '--' + e.topic + '.html'; }
  function pageHref(e) { return encodeURIComponent(pageFile(e)); }

  // one list item (search result or nav file) from <template id="tpl-item">
  function makeItem(e, title, terms) {
    var li = document.getElementById('tpl-item').content.firstElementChild.cloneNode(true);
    li.querySelector('.title').textContent = title;
    fillMeta(li.querySelector('.meta'), e);
    li.querySelector('.desc').textContent = describe(e.text, terms);
    var go = li.querySelector('.go');
    go.href = pageHref(e);
    go.setAttribute('aria-label', 'Open ' + title);
    return li;
  }

  // ---- search --------------------------------------------------------------
  function setupSearch(index) {
    var search = document.getElementById('mrai-search');
    var input = document.getElementById('mrai-q');
    var panes = document.getElementById('mrai-panes');
    var list = document.getElementById('mrai-results');
    var preview = document.getElementById('mrai-preview');
    var count = document.getElementById('mrai-count');
    var MAX = 50;
    var terms = [], results = [], active = -1;

    function select(i) {
      if (i < 0 || i >= results.length) return;
      if (active >= 0) results[active].li.classList.remove('active');
      active = i;
      var r = results[i];
      r.li.classList.add('active');
      r.li.scrollIntoView({ block: 'nearest' });

      var head = document.getElementById('tpl-preview-head').content.cloneNode(true);
      head.querySelector('.crumb').textContent = label(r.e);
      fillMeta(head.querySelector('.meta'), r.e);
      var body = document.createElement('div');
      body.innerHTML = r.e.html; // rendered by markdown-it with html disabled: escaped
      preview.replaceChildren(head, body);

      if (window.hljs) body.querySelectorAll('pre code').forEach(function (c) { hljs.highlightElement(c); });
      highlightTerms(body, terms);
      var first = body.querySelector('mark');
      preview.scrollTop = first
        ? Math.max(0, preview.scrollTop + first.getBoundingClientRect().top -
            preview.getBoundingClientRect().top - 40)
        : 0;
    }

    // collapsing keeps the query and results; focusing the input brings them back
    function setOpen(open) {
      var show = open && terms.length > 0;
      panes.hidden = !show;
      count.hidden = !show;
      root.classList.toggle('searching', show);
    }

    function run() {
      list.replaceChildren();
      preview.replaceChildren();
      results = [];
      active = -1;
      terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      count.textContent = '';
      setOpen(true);
      if (!terms.length) return;

      index.forEach(function (e) {
        if (results.length >= MAX) return;
        var hay = (e.title + ' ' + e.heading + ' ' + e.text).toLowerCase();
        if (!terms.every(function (t) { return hay.indexOf(t) >= 0; })) return;
        var li = makeItem(e, label(e), terms);
        var k = results.length;
        li.querySelector('.pick').addEventListener('click', function () { select(k); });
        list.append(li);
        results.push({ li: li, e: e });
      });
      count.textContent = results.length
        ? results.length + (results.length >= MAX ? '+' : '') + ' match(es) · ↑↓ to browse'
        : 'No matches';
      select(0);
    }

    input.addEventListener('input', run);
    input.addEventListener('focus', function () { setOpen(true); });
    input.addEventListener('keydown', function (ev) {
      if (ev.key === 'ArrowDown') { ev.preventDefault(); select(active + 1); }
      else if (ev.key === 'ArrowUp') { ev.preventDefault(); select(active - 1); }
    });
    document.addEventListener('keydown', function (ev) {
      var focused = search.contains(document.activeElement);
      if (ev.key !== 'Escape' || (!focused && panes.hidden)) return;
      ev.preventDefault(); // also stops type=search from clearing the query
      setOpen(false);
      if (focused) document.activeElement.blur();
    });
    document.addEventListener('mousedown', function (ev) {
      if (!search.contains(ev.target)) setOpen(false);
    });
  }

  // ---- file nav ------------------------------------------------------------
  function setupNav(index) {
    var nav = document.getElementById('mrai-nav');
    var files = document.getElementById('mrai-files');
    var here = decodeURIComponent(location.pathname.split('/').pop());
    var seen = {};
    index.forEach(function (e) { // index is newest first, so the first hit per topic is its latest
      var key = e.ws + '/' + e.topic;
      if (seen[key]) return;
      seen[key] = true;
      var li = makeItem(e, e.title, []);
      if (here === pageFile(e)) li.classList.add('active');
      li.querySelector('.pick').addEventListener('click', function () { location.href = pageHref(e); });
      files.append(li);
    });
    document.getElementById('mrai-nav-toggle').addEventListener('click', function () {
      var next = getComputedStyle(nav).display === 'none' ? 'open' : 'closed';
      root.dataset.nav = next;
      store.set('mrai-nav', next);
    });
  }

  // ---- theme button, code blocks --------------------------------------------
  function setupThemeButton() {
    var btn = document.getElementById('mrai-theme');
    function paint() { btn.textContent = currentTheme() === 'dark' ? '☀' : '☾'; }
    paint();
    btn.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      store.set('mrai-theme', next);
      paint();
    });
  }

  function setupCodeBlocks() {
    if (window.hljs) hljs.highlightAll();
    document.querySelectorAll('article pre').forEach(function (pre) {
      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.addEventListener('click', function () {
        var code = pre.querySelector('code');
        navigator.clipboard.writeText((code || pre).textContent).then(function () {
          btn.textContent = 'Copied!';
          setTimeout(function () { btn.textContent = 'Copy'; }, 2000);
        });
      });
      pre.appendChild(btn);
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    var check = document.getElementById('mrai-asset-check');
    if (check) check.remove();
    var index = window.MRAI_INDEX || [];
    setupThemeButton();
    setupSearch(index);
    setupNav(index);
    setupCodeBlocks();
  });
})();
