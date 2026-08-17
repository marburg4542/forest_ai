/* Service worker for the installed app.
 *
 * This caches the interface, not the science: every measurement happens in the
 * Python process, so the app still needs `python serve.py` running. What the
 * worker buys is an instant cold start — the shell is ~5 MB, almost all of it
 * plotly.min.js — and a readable page instead of a browser error when the
 * server is not up.
 *
 * Bump CACHE_VERSION whenever the precached list changes. */

const CACHE_VERSION = 'v2';
const CACHE = `forest-ai-${CACHE_VERSION}`;

const SHELL = [
  '/',
  '/static/app.css',
  '/static/app.js',
  '/static/vendor/plotly.min.js',
  '/manifest.webmanifest',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // individually, so one missing file cannot fail the whole install
    await Promise.allSettled(SHELL.map(u => cache.add(u)));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Results, figures and uploads are per-session and change on every run.
  // Caching any of it would show one run's numbers under another run's page.
  if (url.pathname.startsWith('/api/')) return;

  // Never let the worker serve itself from cache — the browser has its own
  // update mechanism for this file and a cached copy would stall it.
  if (url.pathname === '/sw.js') return;

  // Navigation: prefer the live server so a restarted app is picked up at once,
  // fall back to the cached shell so the window still opens when it is down.
  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try {
        return await fetch(req);
      } catch (_) {
        return (await caches.match('/')) || Response.error();
      }
    })());
    return;
  }

  // Static assets: serve from cache at once, refresh in the background, so
  // editing app.css does not require clearing site data to see the change.
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(req);
    const network = fetch(req).then((res) => {
      if (res && res.ok) cache.put(req, res.clone());
      return res;
    }).catch(() => null);
    return hit || (await network) || Response.error();
  })());
});
