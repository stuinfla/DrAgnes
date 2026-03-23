/**
 * DrAgnes HuggingFace Classification Proxy -- V2 (secondary model)
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

const DEFAULT_MODEL_2 = "skintaglabs/siglip-skin-lesion-classifier";

/** Read the model name from environment or use the default */
function getModel(): string {
	return env.HF_MODEL_2 || DEFAULT_MODEL_2;
}

function getApiUrl(): string {
	return `https://router.huggingface.co/hf-inference/models/${getModel()}`;
}

export const POST: RequestHandler = async ({ request }) => {
	const formData = await request.formData();
	const imageFile = formData.get("image");

	if (!imageFile || !(imageFile instanceof File || imageFile instanceof Blob)) {
		throw error(400, "No image provided");
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
			body: imageFile,
		});

		if (!response.ok) {
			const text = await response.text();
			console.error(`[dragnes/classify-v2] HF API v2 error: ${response.status} ${text}`);
			throw error(response.status, `Classification service v2 error: ${text}`);
		}

		const results = await response.json();
		return json({ results, model });
	} catch (err) {
		// Re-throw SvelteKit errors
		if (err && typeof err === "object" && "status" in err) {
			throw err;
		}
		console.error("[dragnes/classify-v2] Classification v2 failed:", err);
		throw error(500, "Classification service v2 unavailable");
	}
};
