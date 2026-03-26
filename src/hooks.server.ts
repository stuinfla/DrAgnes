/**
 * SvelteKit Server Hooks -- Security Headers
 *
 * Adds security headers to ALL responses:
 * - CSP, X-Frame-Options, HSTS, nosniff, Referrer-Policy, Permissions-Policy
 *
 * Camera permission is explicitly allowed (required for skin lesion capture).
 */

import type { Handle } from "@sveltejs/kit";

export const handle: Handle = async ({ event, resolve }) => {
	const response = await resolve(event);

	response.headers.set(
		"Content-Security-Policy",
		"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' https://pi.ruv.io; font-src 'self'; frame-ancestors 'none'",
	);
	response.headers.set("X-Frame-Options", "DENY");
	response.headers.set("X-Content-Type-Options", "nosniff");
	response.headers.set("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
	response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
	response.headers.set("Permissions-Policy", "camera=(self), microphone=()");

	return response;
};
