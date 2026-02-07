/**
 * Service Worker for AI Engineering Tutorial
 * Enables offline functionality and caching
 */

const CACHE_NAME = 'ai-tutorial-v1';
const STATIC_CACHE = 'ai-tutorial-static-v1';
const CONTENT_CACHE = 'ai-tutorial-content-v1';

// Files to cache immediately
const STATIC_ASSETS = [
    '/',
    '/static/css/styles.css',
    '/static/js/main.js',
    '/static/js/progress-tracker.js',
    '/static/js/accessibility.js',
    '/static/js/theme-manager.js',
    '/offline.html'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[SW] Installing service worker...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('[SW] Static assets cached');
                return self.skipWaiting();
            })
            .catch((error) => {
                console.error('[SW] Failed to cache static assets:', error);
            })
    );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating service worker...');
    
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => {
                            return name.startsWith('ai-tutorial-') && 
                                   name !== STATIC_CACHE && 
                                   name !== CONTENT_CACHE;
                        })
                        .map((name) => {
                            console.log('[SW] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                console.log('[SW] Service worker activated');
                return self.clients.claim();
            })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    
    // Skip non-GET requests
    if (event.request.method !== 'GET') {
        return;
    }
    
    // Skip external requests
    if (url.origin !== location.origin) {
        return;
    }
    
    // Handle API requests differently (network first)
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(networkFirst(event.request));
        return;
    }
    
    // Handle static assets (cache first)
    if (url.pathname.startsWith('/static/')) {
        event.respondWith(cacheFirst(event.request, STATIC_CACHE));
        return;
    }
    
    // Handle content pages (stale while revalidate)
    if (url.pathname.startsWith('/learn/')) {
        event.respondWith(staleWhileRevalidate(event.request, CONTENT_CACHE));
        return;
    }
    
    // Default strategy: network first with cache fallback
    event.respondWith(networkFirst(event.request));
});

/**
 * Cache First Strategy
 * Try cache, fallback to network
 */
async function cacheFirst(request, cacheName) {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    
    if (cached) {
        return cached;
    }
    
    try {
        const response = await fetch(request);
        if (response.ok) {
            cache.put(request, response.clone());
        }
        return response;
    } catch (error) {
        console.error('[SW] Fetch failed:', error);
        return offlineResponse();
    }
}

/**
 * Network First Strategy
 * Try network, fallback to cache
 */
async function networkFirst(request) {
    try {
        const response = await fetch(request);
        
        if (response.ok) {
            const cache = await caches.open(CONTENT_CACHE);
            cache.put(request, response.clone());
        }
        
        return response;
    } catch (error) {
        console.log('[SW] Network failed, trying cache:', request.url);
        
        const cached = await caches.match(request);
        if (cached) {
            return cached;
        }
        
        return offlineResponse();
    }
}

/**
 * Stale While Revalidate Strategy
 * Return cache immediately, update cache in background
 */
async function staleWhileRevalidate(request, cacheName) {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    
    // Fetch fresh version in background
    const fetchPromise = fetch(request)
        .then((response) => {
            if (response.ok) {
                cache.put(request, response.clone());
            }
            return response;
        })
        .catch(() => null);
    
    // Return cached version immediately if available
    if (cached) {
        return cached;
    }
    
    // Wait for network if no cache
    const response = await fetchPromise;
    if (response) {
        return response;
    }
    
    return offlineResponse();
}

/**
 * Return offline page
 */
async function offlineResponse() {
    const cache = await caches.open(STATIC_CACHE);
    const offlinePage = await cache.match('/offline.html');
    
    if (offlinePage) {
        return offlinePage;
    }
    
    return new Response('You are offline. Please check your connection.', {
        status: 503,
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'text/plain' }
    });
}

// Handle messages from main thread
self.addEventListener('message', (event) => {
    if (event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data.type === 'CACHE_URLS') {
        cacheUrls(event.data.urls);
    }
    
    if (event.data.type === 'CLEAR_CACHE') {
        clearCache(event.data.cacheName);
    }
});

/**
 * Cache specific URLs
 */
async function cacheUrls(urls) {
    const cache = await caches.open(CONTENT_CACHE);
    
    for (const url of urls) {
        try {
            const response = await fetch(url);
            if (response.ok) {
                await cache.put(url, response);
                console.log('[SW] Cached:', url);
            }
        } catch (error) {
            console.error('[SW] Failed to cache:', url, error);
        }
    }
}

/**
 * Clear specific cache
 */
async function clearCache(cacheName) {
    if (cacheName) {
        await caches.delete(cacheName);
        console.log('[SW] Cleared cache:', cacheName);
    } else {
        const cacheNames = await caches.keys();
        await Promise.all(cacheNames.map(name => caches.delete(name)));
        console.log('[SW] Cleared all caches');
    }
}

// Background sync for progress data
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-progress') {
        event.waitUntil(syncProgress());
    }
});

/**
 * Sync progress data when back online
 */
async function syncProgress() {
    console.log('[SW] Syncing progress data...');
    
    // Get queued sync data from IndexedDB or localStorage
    // This would be implemented based on the ProgressTracker's sync queue
    
    try {
        // Notify clients that sync is complete
        const clients = await self.clients.matchAll();
        clients.forEach(client => {
            client.postMessage({ type: 'SYNC_COMPLETE' });
        });
    } catch (error) {
        console.error('[SW] Sync failed:', error);
    }
}
