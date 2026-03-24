/**
 * DrAgnes ONNX Model Cache Service Worker
 *
 * Intercepts fetch requests for /models/*.onnx files and caches them
 * using the Cache API for offline-capable inference.
 *
 * ADR-122 Phase 4
 */

const CACHE_NAME = "dragnes-models-v1";

self.addEventListener("install", () => {
	self.skipWaiting();
});

self.addEventListener("activate", (event) => {
	// Purge old model cache versions on activation
	event.waitUntil(
		caches.keys().then((keys) =>
			Promise.all(
				keys
					.filter((k) => k.startsWith("dragnes-models-") && k !== CACHE_NAME)
					.map((k) => caches.delete(k))
			)
		).then(() => self.clients.claim())
	);
});

self.addEventListener("fetch", (event) => {
	const url = new URL(event.request.url);

	// Only intercept ONNX model requests
	if (!url.pathname.startsWith("/models/") || !url.pathname.endsWith(".onnx")) {
		return;
	}

	event.respondWith(
		caches.open(CACHE_NAME).then(async (cache) => {
			const cached = await cache.match(event.request);
			if (cached) return cached;

			const response = await fetch(event.request);
			if (response.ok) {
				cache.put(event.request, response.clone());
			}
			return response;
		})
	);
});
