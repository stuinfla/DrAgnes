/**
 * Mela Local Model Classification API
 *
 * POST /api/classify-local
 *
 * Runs the custom-trained ViT model locally via ONNX Runtime (Node.js).
 * No Python needed -- works on Vercel and any Node.js host.
 *
 * Falls back to Python subprocess if ONNX model is not available but
 * PyTorch model.safetensors is on disk.
 *
 * Accepts multipart form data with an "image" field containing the lesion JPEG/PNG.
 * Returns classification results in the same format as the HF proxy endpoints.
 */

import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { rateLimit } from "$lib/server/rate-limit";
import { existsSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { writeFileSync, unlinkSync } from "fs";

const CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const ONNX_MODEL_PATH = join(process.cwd(), "scripts/mela-onnx/model.onnx");
const PYTORCH_MODEL_DIR = join(process.cwd(), "scripts/mela-classifier/best");
const CLASSIFY_SCRIPT = join(process.cwd(), "scripts", "classify-image.py");

let ortSession: any = null;

async function getOnnxSession() {
	if (ortSession) return ortSession;

	if (existsSync(ONNX_MODEL_PATH)) {
		try {
			const ort = await import("onnxruntime-node");
			ortSession = await ort.InferenceSession.create(ONNX_MODEL_PATH);
			return ortSession;
		} catch (e) {
			console.warn("[mela/classify-local] ONNX loading failed:", e);
		}
	}
	return null;
}

async function classifyWithOnnx(
	imageBuffer: Buffer,
): Promise<Array<{ label: string; score: number }>> {
	const session = await getOnnxSession();
	if (!session) return [];

	const ort = await import("onnxruntime-node");

	// Use sharp to preprocess image to 224x224 RGB tensor
	const sharp = (await import("sharp")).default;
	const { data } = await sharp(imageBuffer)
		.resize(224, 224)
		.removeAlpha()
		.raw()
		.toBuffer({ resolveWithObject: true });

	// Convert to NCHW float32 tensor with ImageNet normalization
	const mean = [0.485, 0.456, 0.406];
	const std = [0.229, 0.224, 0.225];
	const float32Data = new Float32Array(1 * 3 * 224 * 224);

	for (let c = 0; c < 3; c++) {
		for (let h = 0; h < 224; h++) {
			for (let w = 0; w < 224; w++) {
				const srcIdx = (h * 224 + w) * 3 + c;
				const dstIdx = c * 224 * 224 + h * 224 + w;
				float32Data[dstIdx] = (data[srcIdx] / 255.0 - mean[c]) / std[c];
			}
		}
	}

	const tensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);
	const results = await session.run({ pixel_values: tensor });
	const logits = results.logits.data as Float32Array;

	// Softmax
	const maxLogit = Math.max(...Array.from(logits));
	const exps = Array.from(logits).map((l) => Math.exp(l - maxLogit));
	const sumExp = exps.reduce((a, b) => a + b, 0);
	const probs = exps.map((e) => e / sumExp);

	return CLASS_NAMES.map((name, i) => ({ label: name, score: probs[i] })).sort(
		(a, b) => b.score - a.score,
	);
}

/** Fallback to Python subprocess if ONNX not available */
async function classifyWithPython(
	imageBuffer: Buffer,
): Promise<Array<{ label: string; score: number }>> {
	const { execSync } = await import("child_process");
	const tmpPath = join(tmpdir(), `mela-${Date.now()}.jpg`);
	writeFileSync(tmpPath, imageBuffer);
	try {
		const result = execSync(`python3 "${CLASSIFY_SCRIPT}" "${tmpPath}"`, {
			cwd: process.cwd(),
			timeout: 30000,
			encoding: "utf-8",
			stdio: ["pipe", "pipe", "pipe"],
		});
		return JSON.parse(result.trim());
	} finally {
		try {
			unlinkSync(tmpPath);
		} catch {
			// Best effort cleanup
		}
	}
}

/** Check which backend is available */
function detectBackend(): "onnx" | "python" | "none" {
	if (existsSync(ONNX_MODEL_PATH)) return "onnx";
	if (
		existsSync(join(PYTORCH_MODEL_DIR, "model.safetensors")) &&
		existsSync(CLASSIFY_SCRIPT)
	)
		return "python";
	return "none";
}

/**
 * GET /api/classify-local
 *
 * Health probe -- returns whether the local model is available and which backend.
 */
export const GET: RequestHandler = async () => {
	const backend = detectBackend();

	return json({
		available: backend !== "none",
		backend,
		model: "mela-custom-vit-v1",
		melanomaSensitivity: "95.97%",
		trainedOn: "HAM10000 + ISIC 2019 combined (37,484 images)",
	});
};

/**
 * POST /api/classify-local
 *
 * Classify an image using the local trained ViT model.
 * Tries ONNX first (Node.js native, works on Vercel), falls back to Python.
 */
export const POST: RequestHandler = async ({ request, getClientAddress }) => {
	const limited = rateLimit(getClientAddress(), '/api/classify-local', 30, 60000);
	if (limited) return new Response('Too many requests', { status: 429 });

	const formData = await request.formData();
	const imageFile = formData.get("image");

	if (!imageFile || !(imageFile instanceof Blob)) {
		throw error(400, "No image provided");
	}

	// Security: validate file size
	const MAX_SIZE = 10 * 1024 * 1024; // 10MB
	if (imageFile.size > MAX_SIZE) throw error(413, "Image too large (max 10MB)");
	if (imageFile.size === 0) throw error(400, "Empty file");

	// Security: validate content type
	const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
	if (imageFile.type && !ALLOWED_TYPES.includes(imageFile.type)) {
		throw error(415, `Unsupported image type: ${imageFile.type}. Use JPEG, PNG, or WebP.`);
	}

	const buffer = Buffer.from(await imageFile.arrayBuffer());

	// Security: validate image magic bytes
	if (buffer.length < 4) throw error(400, "File too small to be a valid image");
	const isJPEG = buffer[0] === 0xFF && buffer[1] === 0xD8;
	const isPNG = buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47;
	const isWEBP = buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46;
	if (!isJPEG && !isPNG && !isWEBP) {
		throw error(415, "Invalid image format. File does not contain valid JPEG, PNG, or WebP data.");
	}

	// Try ONNX first, fall back to Python
	let results: Array<{ label: string; score: number }> = [];
	let backend = "onnx";

	results = await classifyWithOnnx(buffer);

	if (results.length === 0) {
		backend = "python";
		try {
			results = await classifyWithPython(buffer);
		} catch (err) {
			console.error("[mela/classify-local] All backends failed:", err);
			throw error(
				503,
				"Classification unavailable -- no model loaded. " +
					"Place ONNX model at scripts/mela-onnx/model.onnx or " +
					"train PyTorch model with: python3 scripts/train-fast.py",
			);
		}
	}

	return json({
		results,
		model: "mela-custom-vit-v1",
		backend,
		melanomaSensitivity: "95.97%",
		trainedOn: "HAM10000 + ISIC 2019 combined (37,484 images)",
	});
};
