// ==========================================================
//  LINGUALENS — Service Worker
//  Caches all app files for offline use
// ==========================================================

const CACHE_NAME = 'lingualens-v1';
const ASSETS = [
    './',
    './index.html',
    './styles.css',
    './dictionary.js',
    './detector.js',
    './app.js',
    './icon-512.png',
    './manifest.json',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap'
];

// TensorFlow + COCO-SSD are large — cache them on first use
const CDN_ASSETS = [
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js'
];

// Install: cache local assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS))
            .then(() => self.skipWaiting())
    );
});

// Activate: clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        ).then(() => self.clients.claim())
    );
});

// Fetch: network-first for CDN (model files), cache-first for local assets
self.addEventListener('fetch', event => {
    const url = event.request.url;

    // CDN resources: try network first, fall back to cache
    if (CDN_ASSETS.some(a => url.includes(a.split('/').pop()))) {
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                    return response;
                })
                .catch(() => caches.match(event.request))
        );
        return;
    }

    // Local assets: cache first, fall back to network
    event.respondWith(
        caches.match(event.request).then(cached => {
            if (cached) return cached;
            return fetch(event.request).then(response => {
                // Cache new successful responses
                if (response.status === 200) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                }
                return response;
            });
        })
    );
});
