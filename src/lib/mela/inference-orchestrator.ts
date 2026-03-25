/**
 * Inference Orchestrator for Mela
 *
 * Routes classification through the local ONNX V2 model (95.97% mel sensitivity).
 * No external API dependencies — runs 100% on device.
 *
 * Strategy:
 *  - auto (default): ONNX model if loaded, else trained-weights + rules fallback
 *  - offline: ONNX only, throws if not loaded
 *
 * ADR-122 / RuVector-native architecture
 */

import type { ClassProbability } from "./types";
import { initOfflineModel, isOfflineModelLoaded, classifyOffline } from "./inference-offline";

export type InferenceStrategy = "auto" | "offline";

export interface InferenceResult {
	/** Sorted class probabilities from the chosen backend. */
	probabilities: ClassProbability[];
	/** Which backend actually produced the result. */
	backend: "onnx-local" | "features-fallback";
	/** Wall-clock inference time in milliseconds. */
	latencyMs: number;
}

/**
 * Classify a skin lesion image using the local ONNX V2 model.
 *
 * @param imageData - Raw RGBA ImageData from a canvas capture.
 * @param strategy  - Routing strategy (default: "auto").
 *
 * Strategy behaviour:
 *  - **auto** (default): Use the ONNX model if loaded, otherwise return null
 *    to signal the caller should use the trained-weights + rules fallback.
 *  - **offline**: Use only the ONNX model; throws if not loaded.
 */
export async function classify(
	imageData: ImageData,
	strategy: InferenceStrategy = "auto",
): Promise<InferenceResult | null> {
	const start = performance.now();

	if (strategy === "offline") {
		if (!isOfflineModelLoaded()) {
			throw new Error(
				"Offline strategy requested but ONNX model is not loaded. " +
				"Call initOfflineModel() first.",
			);
		}
		const probabilities = await classifyOffline(imageData);
		return {
			probabilities,
			backend: "onnx-local",
			latencyMs: Math.round(performance.now() - start),
		};
	}

	// auto: try ONNX, return null if unavailable
	if (isOfflineModelLoaded()) {
		try {
			const probabilities = await classifyOffline(imageData);
			return {
				probabilities,
				backend: "onnx-local",
				latencyMs: Math.round(performance.now() - start),
			};
		} catch {
			// ONNX inference failed; caller will use fallback
			return null;
		}
	}

	// ONNX not loaded — caller handles fallback
	return null;
}

/**
 * Pre-warm the ONNX model. Safe to call multiple times.
 * Downloads and caches the 85MB model via service worker on first call.
 * Returns true when the model is ready, false if unavailable.
 */
export async function warmOfflineModel(): Promise<boolean> {
	return initOfflineModel();
}

/**
 * Whether the ONNX model is loaded and ready.
 */
export { isOfflineModelLoaded };
