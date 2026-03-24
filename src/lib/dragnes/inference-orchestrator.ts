/**
 * Inference Orchestrator for DrAgnes
 *
 * Routes classification requests between:
 *  - offline: ONNX Runtime Web in-browser inference (no network)
 *  - online:  HF API via existing DermClassifier ensemble
 *  - auto:    offline-first with online fallback
 *
 * ADR-122 Phase 5
 */

import type { ClassProbability } from "./types";
import { initOfflineModel, isOfflineModelLoaded, classifyOffline } from "./inference-offline";
import { DermClassifier } from "./classifier";

export type InferenceStrategy = "online" | "offline" | "auto";

export interface InferenceResult {
	/** Sorted class probabilities from the chosen backend. */
	probabilities: ClassProbability[];
	/** Which backend actually produced the result. */
	backend: "onnx-offline" | "hf-online";
	/** Wall-clock inference time in milliseconds. */
	latencyMs: number;
}

/** Lazily-created online classifier (shared across calls). */
let onlineClassifier: DermClassifier | null = null;

function getOnlineClassifier(): DermClassifier {
	if (!onlineClassifier) {
		onlineClassifier = new DermClassifier();
	}
	return onlineClassifier;
}

/**
 * Classify a dermoscopic image using the specified strategy.
 *
 * @param imageData - Raw RGBA ImageData from a canvas capture.
 * @param strategy  - Routing strategy (default: "auto").
 *
 * Strategy behaviour:
 *  - **auto** (default): Use the offline ONNX model if it has been loaded,
 *    otherwise fall back to the online HF API ensemble.
 *  - **online**: Always route through DermClassifier (HF API + local ensemble).
 *  - **offline**: Use only the ONNX model; throws if it has not been loaded.
 */
export async function classify(
	imageData: ImageData,
	strategy: InferenceStrategy = "auto",
): Promise<InferenceResult> {
	const start = performance.now();

	// ---- offline ----
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
			backend: "onnx-offline",
			latencyMs: Math.round(performance.now() - start),
		};
	}

	// ---- auto ----
	if (strategy === "auto" && isOfflineModelLoaded()) {
		try {
			const probabilities = await classifyOffline(imageData);
			return {
				probabilities,
				backend: "onnx-offline",
				latencyMs: Math.round(performance.now() - start),
			};
		} catch {
			// ONNX inference failed; fall through to online
		}
	}

	// ---- online (also the fallback for auto) ----
	const clf = getOnlineClassifier();
	const result = await clf.classify(imageData);
	return {
		probabilities: result.probabilities,
		backend: "hf-online",
		latencyMs: Math.round(performance.now() - start),
	};
}

/**
 * Pre-warm the offline model. Safe to call multiple times.
 * Returns true when the model is ready, false if unavailable.
 */
export async function warmOfflineModel(): Promise<boolean> {
	return initOfflineModel();
}
