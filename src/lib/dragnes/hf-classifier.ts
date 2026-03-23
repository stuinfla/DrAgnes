/**
 * HuggingFace Inference API classifier for skin lesion classification.
 * Uses Anwarkh1/Skin_Cancer-Image_Classification (ViT, 85.8M params)
 * trained on HAM10000 dataset.
 *
 * This module provides:
 * - Direct HF API call (for server-side use)
 * - Label mapping from HF model output to canonical 7-class names
 * - Utility to convert ImageData to Blob for API transport
 */

const HF_MODEL = "Anwarkh1/Skin_Cancer-Image_Classification";
const HF_API_URL = `https://router.huggingface.co/hf-inference/models/${HF_MODEL}`;

/** Map HF model labels to our canonical class names */
const LABEL_MAP: Record<string, string> = {
	// Short HAM10000 abbreviations
	"akiec": "akiec",
	"bcc": "bcc",
	"bkl": "bkl",
	"df": "df",
	"mel": "mel",
	"nv": "nv",
	"vasc": "vasc",
	// Full names the model might return
	"Actinic keratoses": "akiec",
	"Basal cell carcinoma": "bcc",
	"Benign keratosis-like lesions": "bkl",
	"Dermatofibroma": "df",
	"Melanoma": "mel",
	"Melanocytic nevi": "nv",
	"Vascular lesions": "vasc",
	// Singular / alternate forms
	"actinic keratosis": "akiec",
	"basal cell carcinoma": "bcc",
	"benign keratosis": "bkl",
	"dermatofibroma": "df",
	"melanoma": "mel",
	"melanocytic nevus": "nv",
	"vascular lesion": "vasc",
};

export { LABEL_MAP, HF_MODEL, HF_API_URL };

export interface HFClassificationResult {
	label: string;
	score: number;
}

/**
 * Classify a skin lesion image using the HuggingFace Inference API.
 *
 * @param imageBlob - Image as Blob
 * @param apiKey - HuggingFace API token (optional for free tier, needed for higher rate limits)
 * @returns Array of {label, score} sorted by score descending
 */
export async function classifyWithHF(
	imageBlob: Blob,
	apiKey?: string,
): Promise<HFClassificationResult[]> {
	const headers: Record<string, string> = {};
	if (apiKey) {
		headers["Authorization"] = `Bearer ${apiKey}`;
	}

	const response = await fetch(HF_API_URL, {
		method: "POST",
		headers,
		body: imageBlob,
	});

	if (!response.ok) {
		const text = await response.text();
		throw new Error(`HF API error ${response.status}: ${text}`);
	}

	const results: Array<{ label: string; score: number }> = await response.json();
	return results;
}

/**
 * Convert HF results to our canonical 7-class probability map.
 * Normalizes scores to sum to 1.
 */
export function mapHFResultsToClasses(
	results: HFClassificationResult[],
): Record<string, number> {
	const probs: Record<string, number> = {
		akiec: 0, bcc: 0, bkl: 0, df: 0, mel: 0, nv: 0, vasc: 0,
	};

	for (const r of results) {
		const canonical = LABEL_MAP[r.label] || LABEL_MAP[r.label.toLowerCase()];
		if (canonical && canonical in probs) {
			probs[canonical] = r.score;
		}
	}

	// Normalize to sum to 1
	const total = Object.values(probs).reduce((a, b) => a + b, 0);
	if (total > 0) {
		for (const cls of Object.keys(probs)) {
			probs[cls] /= total;
		}
	}

	return probs;
}

/**
 * Convert a canvas ImageData to a JPEG Blob for sending to the API.
 */
export function imageDataToBlob(imageData: ImageData): Promise<Blob> {
	return new Promise((resolve, reject) => {
		const canvas = document.createElement("canvas");
		canvas.width = imageData.width;
		canvas.height = imageData.height;
		const ctx = canvas.getContext("2d");
		if (!ctx) {
			reject(new Error("No canvas context"));
			return;
		}
		ctx.putImageData(imageData, 0, 0);
		canvas.toBlob(
			(blob) => {
				if (blob) resolve(blob);
				else reject(new Error("Failed to create blob"));
			},
			"image/jpeg",
			0.95,
		);
	});
}
