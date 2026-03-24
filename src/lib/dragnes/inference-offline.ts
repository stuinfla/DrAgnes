/**
 * Offline ONNX Runtime Web inference for DrAgnes.
 *
 * Loads a quantized (int8) ONNX model in the browser via onnxruntime-web
 * and runs ViT classification without any server round-trip.
 * Falls back gracefully when the package or model is unavailable.
 *
 * ADR-122 Phase 3
 */

import type { ClassProbability, LesionClass } from "./types";
import { LESION_LABELS } from "./types";
import { resizeBilinear } from "./preprocessing";

/** HAM10000 7-class canonical order matching the ONNX model output head. */
const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/** Path served from static/models/ via the Service Worker cache. */
const MODEL_URL = "/models/dragnes-v2-int8.onnx";

/** ImageNet channel means (RGB). */
const MEAN: [number, number, number] = [0.485, 0.456, 0.406];

/** ImageNet channel standard deviations (RGB). */
const STD: [number, number, number] = [0.229, 0.224, 0.225];

/** onnxruntime-web InferenceSession (lazy-loaded). */
let session: any = null;

/** Reference to the dynamically imported ort module. */
let ort: any = null;

/**
 * Initialise the offline ONNX model.
 *
 * Dynamically imports onnxruntime-web (it may not be installed) and
 * creates an InferenceSession from the quantized model served at MODEL_URL.
 *
 * @returns true when the model is ready, false when onnxruntime-web or the
 *          model file is unavailable.
 */
export async function initOfflineModel(): Promise<boolean> {
	if (session) return true;

	try {
		const moduleName = "onnxruntime-web";
		ort = await import(/* @vite-ignore */ moduleName);

		// Prefer WASM backend (works everywhere), fall back to WebGL
		session = await ort.InferenceSession.create(MODEL_URL, {
			executionProviders: ["wasm"],
			graphOptimizationLevel: "all",
		});

		return true;
	} catch {
		session = null;
		ort = null;
		return false;
	}
}

/**
 * Whether the offline ONNX model is loaded and ready for inference.
 */
export function isOfflineModelLoaded(): boolean {
	return session !== null;
}

/**
 * Run offline classification on raw RGBA ImageData.
 *
 * Pipeline:
 *  1. Resize to 224x224 via bilinear interpolation
 *  2. Convert to NCHW float32 tensor with ImageNet normalisation
 *  3. Run ONNX inference
 *  4. Softmax the logits
 *  5. Return sorted ClassProbability array
 *
 * @throws Error if the offline model has not been loaded.
 */
export async function classifyOffline(imageData: ImageData): Promise<ClassProbability[]> {
	if (!session || !ort) {
		throw new Error("Offline ONNX model is not loaded. Call initOfflineModel() first.");
	}

	// 1. Resize to 224x224
	const resized = resizeBilinear(imageData, 224, 224);

	// 2. Build NCHW float32 tensor [1, 3, 224, 224]
	const { data, width, height } = resized;
	const channelSize = width * height;
	const tensorData = new Float32Array(3 * channelSize);

	for (let i = 0; i < channelSize; i++) {
		const px = i * 4; // RGBA offset
		tensorData[i] = (data[px] / 255 - MEAN[0]) / STD[0];
		tensorData[channelSize + i] = (data[px + 1] / 255 - MEAN[1]) / STD[1];
		tensorData[2 * channelSize + i] = (data[px + 2] / 255 - MEAN[2]) / STD[2];
	}

	const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, 224, 224]);

	// 3. Run inference
	const feeds: Record<string, any> = { pixel_values: inputTensor };
	const results = await session.run(feeds);

	// The model may expose logits under different output names
	const outputKey = session.outputNames?.[0] ?? "logits";
	const logits: Float32Array = results[outputKey].data as Float32Array;

	// 4. Softmax
	const probs = softmax(logits);

	// 5. Map to ClassProbability and sort descending
	const output: ClassProbability[] = CLASSES.map((cls, i) => ({
		className: cls,
		probability: probs[i],
		label: LESION_LABELS[cls],
	}));

	output.sort((a, b) => b.probability - a.probability);
	return output;
}

/** Numerically stable softmax over a Float32Array. */
function softmax(logits: Float32Array): number[] {
	const max = logits.reduce((m, v) => Math.max(m, v), -Infinity);
	const exps = Array.from(logits, (v) => Math.exp(v - max));
	const sum = exps.reduce((s, e) => s + e, 0);
	return exps.map((e) => e / sum);
}
