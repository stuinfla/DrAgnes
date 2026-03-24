/**
 * Measurement Orchestrator -- ADR-121 Phase 4
 *
 * Combines three calibration strategies in priority order:
 *   1. Physical connector reference (USB-C / Lightning / USB-A)  -> high confidence
 *   2. Skin texture frequency analysis                           -> medium confidence
 *   3. Dermoscope field-of-view estimate                         -> low confidence
 *
 * Exposes a single `measureLesion()` entry point for the ABCDE pipeline.
 */

import type { BodyLocation } from "./types";
import { detectConnector } from "./measurement-connector";
import type { ConnectorDetection } from "./measurement-connector";
import { measureFromSkinTexture } from "./measurement-texture";
import type { TextureMeasurement } from "./measurement-texture";
import { estimateDiameterMm } from "./image-analysis";

// ============================================================
// Public types
// ============================================================

export interface LesionMeasurement {
	/** Estimated lesion diameter in millimetres */
	diameterMm: number;
	/** Confidence tier based on calibration method */
	confidence: "high" | "medium" | "low";
	/** Which calibration strategy produced the measurement */
	method: "connector" | "texture" | "estimate";
	/** Calibrated pixels-per-mm ratio used for the conversion */
	pixelsPerMm: number;
	/** Human-readable explanation of the measurement and its accuracy */
	details: string;
}

// ============================================================
// Confidence thresholds
// ============================================================

/** Minimum connector detection confidence to trust the calibration */
const CONNECTOR_CONFIDENCE_MIN = 0.6;
/** Minimum texture detection confidence to trust the calibration */
const TEXTURE_CONFIDENCE_MIN = 0.4;

// ============================================================
// Diameter conversion
// ============================================================

/**
 * Convert a lesion area (in pixels) to a diameter (in mm) given a
 * known pixels-per-mm ratio. Assumes a roughly circular lesion.
 *
 *   radius_px  = sqrt(area_px / pi)
 *   diameter_mm = 2 * radius_px / px_per_mm
 */
function areaToDiameterMm(lesionAreaPixels: number, pixelsPerMm: number): number {
	if (pixelsPerMm <= 0) return 0;
	const radiusPx = Math.sqrt(lesionAreaPixels / Math.PI);
	const diameter = (2 * radiusPx) / pixelsPerMm;
	return Math.round(diameter * 10) / 10; // one decimal place
}

// ============================================================
// Safety annotation
// ============================================================

/** The ABCDE "D" clinical threshold in mm */
const D_THRESHOLD_MM = 6;
/** Range around the threshold where uncertainty is clinically relevant */
const UNCERTAIN_LOW_MM = 4;
const UNCERTAIN_HIGH_MM = 8;

const SAFETY_WARNING =
	"Measurement uncertain near the 6mm clinical threshold. " +
	"See a dermatologist if this spot is growing.";

/**
 * If confidence is low and the measured diameter straddles the 6 mm
 * clinical threshold (4-8 mm), append a safety warning.
 */
function appendSafetyWarning(
	details: string,
	diameterMm: number,
	confidence: "high" | "medium" | "low",
): string {
	if (
		confidence === "low" &&
		diameterMm >= UNCERTAIN_LOW_MM &&
		diameterMm <= UNCERTAIN_HIGH_MM
	) {
		return `${details} ${SAFETY_WARNING}`;
	}
	return details;
}

// ============================================================
// Main orchestrator
// ============================================================

/**
 * Measure a skin lesion using the best available calibration method.
 *
 * Priority:
 *   1. Connector detection (high confidence, +/-0.5 mm)
 *   2. Skin texture analysis (medium confidence, +/-2-3 mm)
 *   3. Dermoscope FOV estimate (low confidence, rough)
 *
 * @param imageData        - Full RGBA image from the camera / dermoscope
 * @param lesionAreaPixels - Segmented lesion area in pixels
 * @param bodyLocation     - Body site (used for texture spacing lookup)
 * @returns LesionMeasurement with diameter, confidence, method, and details
 */
export function measureLesion(
	imageData: ImageData,
	lesionAreaPixels: number,
	bodyLocation: BodyLocation,
): LesionMeasurement {
	// ----------------------------------------------------------
	// Strategy 1: Physical connector reference
	// ----------------------------------------------------------
	const connector: ConnectorDetection = detectConnector(imageData);

	if (connector.detected && connector.confidence > CONNECTOR_CONFIDENCE_MIN) {
		const diameterMm = areaToDiameterMm(lesionAreaPixels, connector.pixelsPerMm);
		const connectorLabel = connector.type
			? connector.type.toUpperCase()
			: "connector";

		const details = `Measured using ${connectorLabel} reference (\u00B10.5mm)`;
		return {
			diameterMm,
			confidence: "high",
			method: "connector",
			pixelsPerMm: connector.pixelsPerMm,
			details,
		};
	}

	// ----------------------------------------------------------
	// Strategy 2: Skin texture frequency analysis
	// ----------------------------------------------------------
	const texture: TextureMeasurement = measureFromSkinTexture(imageData, bodyLocation);

	if (texture.detected && texture.confidence > TEXTURE_CONFIDENCE_MIN) {
		const diameterMm = areaToDiameterMm(lesionAreaPixels, texture.pixelsPerMm);
		let details = `Estimated from skin texture analysis (\u00B12-3mm)`;
		details = appendSafetyWarning(details, diameterMm, "medium");

		return {
			diameterMm,
			confidence: "medium",
			method: "texture",
			pixelsPerMm: texture.pixelsPerMm,
			details,
		};
	}

	// ----------------------------------------------------------
	// Strategy 3: Dermoscope field-of-view fallback
	// ----------------------------------------------------------
	const fallbackDiameter = estimateDiameterMm(lesionAreaPixels, imageData.width);

	// Reverse-engineer pixelsPerMm from the fallback estimate so the
	// interface stays consistent. estimateDiameterMm assumes a 25 mm
	// field of view at 10x, giving pxPerMm = imageWidth / 25.
	const fallbackPxPerMm = imageData.width / 25;

	let details =
		"Rough estimate \u2014 place a USB-C cable next to the spot for accurate measurement";
	details = appendSafetyWarning(details, fallbackDiameter, "low");

	return {
		diameterMm: fallbackDiameter,
		confidence: "low",
		method: "estimate",
		pixelsPerMm: Math.round(fallbackPxPerMm * 100) / 100,
		details,
	};
}
