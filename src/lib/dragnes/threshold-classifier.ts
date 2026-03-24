/**
 * Threshold-based classification for DrAgnes (ADR-123).
 *
 * Instead of plain argmax (highest probability wins), this applies
 * per-class optimal thresholds derived from ROC analysis on HAM10000
 * validation data.  Three modes are available:
 *
 *  - "default"   -- ADR-123 optimal thresholds (best Youden J per class)
 *  - "screening" -- lowered thresholds for malignant classes to maximise
 *                   cancer sensitivity (fewer missed cancers, more false alarms)
 *  - "triage"    -- same as default; explicit alias for clinical workflows
 *
 * @module threshold-classifier
 */

import type { ClassProbability, LesionClass } from "./types";

// ---------------------------------------------------------------------------
// Threshold tables
// ---------------------------------------------------------------------------

/** Optimal thresholds from ADR-123 ROC analysis (Youden J maximised). */
const OPTIMAL_THRESHOLDS: Record<LesionClass, number> = {
	akiec: 0.0586,
	bcc: 0.1454,
	bkl: 0.2913,
	df: 0.5114,
	mel: 0.6204,
	nv: 0.068,
	vasc: 0.782,
};

/** Screening mode: maximise cancer sensitivity (lower malignant thresholds). */
const SCREENING_THRESHOLDS: Record<LesionClass, number> = {
	akiec: 0.03,
	bcc: 0.08,
	bkl: 0.20,
	df: 0.30,
	mel: 0.25,
	nv: 0.15,
	vasc: 0.50,
};

/** Triage mode: balanced sensitivity / specificity (same as optimal). */
const TRIAGE_THRESHOLDS: Record<LesionClass, number> = {
	akiec: 0.0586,
	bcc: 0.1454,
	bkl: 0.2913,
	df: 0.5114,
	mel: 0.6204,
	nv: 0.068,
	vasc: 0.782,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export type ThresholdMode = "screening" | "triage" | "default";

/** Return the threshold table for the requested mode. */
export function getThresholds(mode: ThresholdMode): Record<LesionClass, number> {
	if (mode === "screening") return SCREENING_THRESHOLDS;
	if (mode === "triage") return TRIAGE_THRESHOLDS;
	return OPTIMAL_THRESHOLDS;
}

/**
 * Apply per-class thresholds to an existing probability distribution.
 *
 * Algorithm:
 *  1. Filter to classes whose probability >= their threshold.
 *  2. Among those, pick the highest-probability class.
 *  3. If no class exceeds its threshold, fall back to argmax.
 *
 * @param probabilities  - The raw probability array from the classifier.
 * @param mode           - Which threshold table to use.
 * @returns The re-ranked result with the chosen top class and mode metadata.
 */
export function applyThresholds(
	probabilities: ClassProbability[],
	mode: ThresholdMode = "default",
): {
	topClass: LesionClass;
	confidence: number;
	probabilities: ClassProbability[];
	thresholdMode: ThresholdMode;
} {
	const thresholds = getThresholds(mode);

	// Classes that exceed their per-class threshold
	const aboveThreshold = probabilities.filter(
		(p) => p.probability >= (thresholds[p.className] ?? 0),
	);

	// If nothing exceeds any threshold, fall back to plain argmax
	const candidates = aboveThreshold.length > 0 ? aboveThreshold : probabilities;
	const sorted = [...candidates].sort((a, b) => b.probability - a.probability);

	return {
		topClass: sorted[0].className,
		confidence: sorted[0].probability,
		probabilities: [...probabilities].sort((a, b) => b.probability - a.probability),
		thresholdMode: mode,
	};
}
