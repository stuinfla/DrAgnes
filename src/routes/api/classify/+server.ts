/**
 * Mela HuggingFace Classification Proxy
 *
 * POST /api/classify
 *
 * Proxies image classification requests to the HuggingFace Inference API
 * so the API key is never exposed to the browser. Accepts multipart form
 * data with an "image" field containing the lesion JPEG.
 *
 * Uses Anwarkh1/Skin_Cancer-Image_Classification (ViT, 85.8M params,
 * fine-tuned on HAM10000).
 */

import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { env } from "$env/dynamic/private";

const DEFAULT_MODEL_1 = "stuartkerr/mela-classifier";

/** Read the model name from environment or use the default */
function getModel(): string {
	return env.HF_MODEL_1 || DEFAULT_MODEL_1;
}

function getApiUrl(): string {
	return `https://router.huggingface.co/hf-inference/models/${getModel()}`;
}

/** Read the HF API key from environment variables */
function getApiKey(): string | undefined {
	return env.HF_TOKEN || env.HUGGINGFACE_TOKEN;
}

export const POST: RequestHandler = async ({ request }) => {
	const formData = await request.formData();
	const imageFile = formData.get("image");

	if (!imageFile || !(imageFile instanceof File || imageFile instanceof Blob)) {
		throw error(400, "No image provided");
	}

	// Security: validate file size (max 10MB) and content type
	const MAX_SIZE = 10 * 1024 * 1024;
	if (imageFile.size > MAX_SIZE) {
		throw error(413, "Image too large (max 10MB)");
	}
	if (imageFile.size === 0) {
		throw error(400, "Empty file");
	}
	const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
	if (imageFile.type && !validTypes.includes(imageFile.type)) {
		throw error(415, `Unsupported image type: ${imageFile.type}. Use JPEG, PNG, WebP, or BMP.`);
	}

	const apiKey = getApiKey();
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
			body: imageFile,
		});

		if (!response.ok) {
			const text = await response.text();
			console.error(`[mela/classify] HF API error: ${response.status} ${text}`);
			throw error(response.status, `Classification service error: ${text}`);
		}

		const results = await response.json();
		return json({ results, model });
	} catch (err) {
		// Re-throw SvelteKit errors
		if (err && typeof err === "object" && "status" in err) {
			throw err;
		}
		console.error("[mela/classify] Classification failed:", err);
		throw error(500, "Classification service unavailable");
	}
};
