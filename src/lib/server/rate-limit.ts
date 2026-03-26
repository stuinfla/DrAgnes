/**
 * Simple In-Memory Rate Limiter
 *
 * NOTE: On Vercel serverless, each function instance has its own memory.
 * The rate limit state resets on cold starts and is not shared across
 * instances. This provides basic protection against sustained abuse from
 * a single client but is NOT a substitute for a distributed rate limiter
 * (e.g., Vercel KV, Upstash Redis) in production at scale.
 */

interface RateLimitEntry {
	count: number;
	resetAt: number;
}

const store = new Map<string, RateLimitEntry>();

// Periodically clean up expired entries to prevent memory leaks
const CLEANUP_INTERVAL_MS = 60_000;
let lastCleanup = Date.now();

function cleanup(): void {
	const now = Date.now();
	if (now - lastCleanup < CLEANUP_INTERVAL_MS) return;
	lastCleanup = now;
	for (const [key, entry] of store) {
		if (now > entry.resetAt) {
			store.delete(key);
		}
	}
}

/**
 * Check whether a request from the given IP should be rate-limited.
 *
 * @param ip         - Client IP address (from event.getClientAddress())
 * @param endpoint   - Logical endpoint name (used as namespace in the store key)
 * @param maxRequests - Maximum requests allowed in the window
 * @param windowMs   - Time window in milliseconds (default: 60 000 = 1 minute)
 * @returns null if allowed, or an object with retryAfterMs if rate-limited
 */
export function rateLimit(
	ip: string,
	endpoint: string,
	maxRequests: number,
	windowMs: number = 60_000,
): { retryAfterMs: number } | null {
	cleanup();

	const key = `${endpoint}:${ip}`;
	const now = Date.now();
	const entry = store.get(key);

	if (!entry || now > entry.resetAt) {
		store.set(key, { count: 1, resetAt: now + windowMs });
		return null;
	}

	entry.count++;
	if (entry.count > maxRequests) {
		return { retryAfterMs: entry.resetAt - now };
	}

	return null;
}
