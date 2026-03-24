import { json } from "@sveltejs/kit";
import { DRAGNES_CONFIG } from "$lib/dragnes/config";
import { existsSync } from "node:fs";
import path from "node:path";

/**
 * Resolve the project root (where package.json lives).
 * process.cwd() is the project root in both dev and production.
 */
const PROJECT_ROOT = process.cwd();

const MODEL_DIR = path.join(PROJECT_ROOT, "scripts", "dragnes-classifier", "best");

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
		version: DRAGNES_CONFIG.modelVersion,
		backbone: DRAGNES_CONFIG.cnnBackbone,
		classes: DRAGNES_CONFIG.classes.length,
		customModel: customModelAvailable,
		customModelSensitivity: customModelAvailable ? "95.97% melanoma (external ISIC 2019)" : null,
		customModelId: customModelAvailable ? "dragnes-custom-vit-v1" : null,
		privacy: {
			dpEpsilon: DRAGNES_CONFIG.privacy.dpEpsilon,
			kAnonymity: DRAGNES_CONFIG.privacy.kAnonymity,
		},
		timestamp: new Date().toISOString(),
	});
}
