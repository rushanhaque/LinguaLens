/**
 * sw.js — Offline support.
 *
 * Two strategies:
 *   • App shell (HTML/CSS/JS) — stale-while-revalidate, so a returning user
 *     gets an instant paint and the next visit gets the update.
 *   • TensorFlow + the model weights — cache-first and never revalidated;
 *     they are versioned by URL and are far too large to re-fetch casually.
 */

const VERSION = 'v2.0.0';
const SHELL_CACHE = `lingualens-shell-${VERSION}`;
const VENDOR_CACHE = 'lingualens-vendor';   // survives shell upgrades on purpose
const MODEL_CACHE = 'lingualens-model';

const SHELL = [
  './',
  './index.html',
  './manifest.json',
  './icon-512.png',
  './styles/tokens.css',
  './styles/base.css',
  './styles/components.css',
  './styles/views.css',
  './src/app.js',
  './src/version.js',
  './src/data/languages.js',
  './src/data/dictionary.js',
  './src/data/achievements.js',
  './src/core/store.js',
  './src/core/tracker.js',
  './src/core/camera.js',
  './src/core/speech.js',
  './src/core/srs.js',
  './src/core/feedback.js',
  './src/ui/icons.js',
  './src/ui/kit.js',
  './src/views/camera.js',
  './src/views/learn.js',
  './src/views/progress.js',
  './src/views/settings.js',
  './src/views/onboarding.js',
  './src/views/wordSheet.js'
];

const isVendor = (url) => url.hostname === 'cdn.jsdelivr.net';
// COCO-SSD weights are served from Google's model hosts.
const isModel = (url) =>
  /(^|\.)tfhub\.dev$/.test(url.hostname) ||
  /storage\.googleapis\.com$/.test(url.hostname) ||
  url.pathname.includes('model.json') ||
  /group\d+-shard/.test(url.pathname);

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    // addAll fails the whole install if any single file 404s; add individually.
    await Promise.all(SHELL.map((url) =>
      cache.add(new Request(url, { cache: 'reload' })).catch(() => {})));
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys
      .filter((k) => k.startsWith('lingualens-shell-') && k !== SHELL_CACHE)
      .map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;

  if (isModel(url)) { event.respondWith(cacheFirst(req, MODEL_CACHE)); return; }
  if (isVendor(url)) { event.respondWith(cacheFirst(req, VENDOR_CACHE)); return; }
  if (url.origin !== location.origin) return;    // leave anything else alone

  // Navigations always resolve to the shell so deep links work offline.
  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try {
        const fresh = await fetch(req);
        (await caches.open(SHELL_CACHE)).put('./index.html', fresh.clone());
        return fresh;
      } catch {
        return (await caches.match('./index.html')) ||
               (await caches.match('./')) ||
               new Response('Offline', { status: 503, statusText: 'Offline' });
      }
    })());
    return;
  }

  event.respondWith(staleWhileRevalidate(req, SHELL_CACHE));
});

async function cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(req);
  if (hit) return hit;
  try {
    const res = await fetch(req);
    // Opaque cross-origin responses are still worth storing for offline replay.
    if (res && (res.ok || res.type === 'opaque')) cache.put(req, res.clone());
    return res;
  } catch {
    return hit || new Response('', { status: 504, statusText: 'Offline' });
  }
}

async function staleWhileRevalidate(req, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(req);
  const network = fetch(req)
    .then((res) => { if (res && res.ok) cache.put(req, res.clone()); return res; })
    .catch(() => null);
  return hit || (await network) || new Response('', { status: 504, statusText: 'Offline' });
}
