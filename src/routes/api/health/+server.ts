import { json } from "@sveltejs/kit";
import { MELA_CONFIG } from "$lib/mela/config";
import { existsSync } from "node:fs";
import path from "node:path";

/**
 * Resolve the project root (where package.json lives).
 * process.cwd() is the project root in both dev and production.
 */
const PROJECT_ROOT = process.cwd();

const MODEL_DIR = path.join(PROJECT_ROOT, "scripts", "mela-classifier", "best");

function isCustomModelAvailable(): boolean {
	return (
		existsSync(path.join(MODEL_DIR, "model.safetensors")) &&
		existsSync(path.join(MODEL_DIR, "config.json"))
	);
}

export async function GET() {
	const customModelAvailable = isCustomModelAvailable();

	return json({
		status: "ok",
		version: MELA_CONFIG.modelVersion,
		backbone: MELA_CONFIG.cnnBackbone,
		classes: MELA_CONFIG.classes.length,
		customModel: customModelAvailable,
		customModelSensitivity: customModelAvailable ? "95.97% melanoma (external ISIC 2019)" : null,
		customModelId: customModelAvailable ? "mela-custom-vit-v1" : null,
		privacy: {
			dpEpsilon: MELA_CONFIG.privacy.dpEpsilon,
			kAnonymity: MELA_CONFIG.privacy.kAnonymity,
		},
		timestamp: new Date().toISOString(),
	});
}
