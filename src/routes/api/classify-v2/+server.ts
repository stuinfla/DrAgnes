/**
 * Mela HuggingFace Classification Proxy -- V2 (secondary model)
 *
 * POST /api/classify-v2
 *
 * Proxies image classification requests to the HuggingFace Inference API
 * so the API key is never exposed to the browser. Accepts multipart form
 * data with an "image" field containing the lesion JPEG.
 *
 * Uses skintaglabs/siglip-skin-lesion-classifier (SigLIP 400M, MIT license,
 * built by a dermatology company). Replaced actavkid model (HTTP 410 removed).
 */

import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { env } from "$env/dynamic/private";
import { rateLimit } from "$lib/server/rate-limit";

const DEFAULT_MODEL_2 = "skintaglabs/siglip-skin-lesion-classifier";

/** Read the model name from environment or use the default */
function getModel(): string {
	return env.HF_MODEL_2 || DEFAULT_MODEL_2;
}

function getApiUrl(): string {
	return `https://router.huggingface.co/hf-inference/models/${getModel()}`;
}

export const POST: RequestHandler = async ({ request, getClientAddress }) => {
	const limited = rateLimit(getClientAddress(), '/api/classify-v2', 30, 60000);
	if (limited) return new Response('Too many requests', { status: 429 });

	const formData = await request.formData();
	const imageFile = formData.get("image");

	if (!imageFile || typeof imageFile === 'string') {
		throw error(400, "No image provided");
	}

	// Security: validate file size and content type
	const MAX_SIZE = 10 * 1024 * 1024;
	if (imageFile.size > MAX_SIZE) throw error(413, "Image too large (max 10MB)");
	if (imageFile.size === 0) throw error(400, "Empty file");
	const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
	if (imageFile.type && !validTypes.includes(imageFile.type)) {
		throw error(415, `Unsupported image type: ${imageFile.type}`);
	}

	// Security: validate image magic bytes
	const buf = new Uint8Array(await imageFile.arrayBuffer());
	if (buf.length < 4) throw error(400, "File too small to be a valid image");
	const isJPEG = buf[0] === 0xFF && buf[1] === 0xD8 && buf[2] === 0xFF;
	const isPNG = buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4E && buf[3] === 0x47;
	const isWEBP = buf[0] === 0x52 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x46;
	if (!isJPEG && !isPNG && !isWEBP) {
		throw error(400, "Invalid image format");
	}

	const apiKey = env.HF_TOKEN || env.HUGGINGFACE_TOKEN;
	const headers: Record<string, string> = {};
	if (apiKey) {
		headers["Authorization"] = `Bearer ${apiKey}`;
	}

	const model = getModel();
	const apiUrl = getApiUrl();

	try {
		const response = await fetch(apiUrl, {
			method: "POST",
			headers,
			body: buf,
		});

		if (!response.ok) {
			const text = await response.text();
			console.error(`[mela/classify-v2] HF API v2 error: ${response.status} ${text}`);
			throw error(502, 'Classification service temporarily unavailable');
		}

		const results = await response.json();
		return json({ results, model });
	} catch (err) {
		// Re-throw SvelteKit errors
		if (err && typeof err === "object" && "status" in err) {
			throw err;
		}
		console.error("[mela/classify-v2] Classification v2 failed:", err);
		throw error(500, "Classification service v2 unavailable");
	}
};
