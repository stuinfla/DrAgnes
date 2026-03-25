/**
 * Measurement utilities: physical diameter estimation, risk level
 * computation, and lesion presence detection (safety gate).
 */

import type { LesionPresenceResult } from "./types";
import { detectSpots as detectSpotsImpl } from "../spot-detector";

/**
 * Estimate lesion diameter in millimeters from pixel area.
 *
 * Assumes a standard dermoscope field of view of ~25mm at 10x
 * and that the image represents the full field.
 *
 * @param areaPixels - Lesion area in pixels
 * @param imageWidth - Image width in pixels
 * @param magnification - Dermoscope magnification (default 10x)
 * @returns Estimated diameter in millimeters
 */
export function estimateDiameterMm(
	areaPixels: number,
	imageWidth: number,
	magnification: number = 10,
): number {
	// Standard dermoscope: ~25mm field of view at 10x
	const fieldOfViewMm = 25 / (magnification / 10);
	const pxPerMm = imageWidth / fieldOfViewMm;
	const radiusPx = Math.sqrt(areaPixels / Math.PI);
	const diameterMm = (2 * radiusPx) / pxPerMm;
	return Math.round(diameterMm * 10) / 10;
}

/**
 * Compute risk level from real ABCDE feature values.
 *
 * Based on the ABCDE melanoma screening rule:
 * - Asymmetry > 1 = +1 risk point
 * - Border > 4 irregular segments = +1 risk point
 * - Color count >= 3 = +1 risk point
 * - Diameter > 6mm = +1 risk point
 *
 * @returns Risk level: low, moderate, high, or critical
 */
export function computeRiskLevel(
	asymmetry: number,
	borderScore: number,
	colorCount: number,
	diameterMm: number = 0,
): "low" | "moderate" | "high" | "critical" {
	let riskPoints = 0;

	if (asymmetry >= 1) riskPoints++;
	if (asymmetry >= 2) riskPoints++;
	if (borderScore >= 4) riskPoints++;
	if (borderScore >= 6) riskPoints++;
	if (colorCount >= 3) riskPoints++;
	if (colorCount >= 5) riskPoints++;
	if (diameterMm > 6) riskPoints++;

	if (riskPoints <= 1) return "low";
	if (riskPoints <= 3) return "moderate";
	if (riskPoints <= 5) return "high";
	return "critical";
}

/**
 * Detect whether the image actually contains a skin lesion.
 * Returns a confidence score (0-1) that a lesion is present.
 *
 * This is a SAFETY GATE: normal skin (e.g. a hand with no moles)
 * must NOT proceed to classification, which would spuriously
 * label it as melanoma because the model was trained only on
 * lesion images.
 *
 * Multi-layer gating (strictest first):
 * 1. Overall color uniformity -- reject images with very low variance (plain skin)
 * 2. Center-vs-periphery luminance -- a real lesion has a DARK CENTER
 * 3. Segmentation area ratio -- must be 1-70% of image
 * 4. Color contrast (Euclidean RGB) -- raised to 35 from old 15 to reject shadows/veins
 * 5. Compactness -- real lesions are roughly round/oval, not linear shadows
 */
export function detectLesionPresence(imageData: ImageData): LesionPresenceResult {
	// Delegate to the two-pass hybrid spot detector (see spot-detector.ts).
	// This replaces the previous five-gate approach that missed small/off-center moles.
	const spot = detectSpotsImpl(imageData);

	return {
		hasLesion: spot.hasSpot,
		confidence: spot.confidence,
		reason: spot.reason,
	};
}
