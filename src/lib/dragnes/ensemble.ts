/**
 * ADR-125: V1+V2 Dual Model Ensemble
 *
 * Weighted-average ensemble of two HuggingFace ViT classifiers with a
 * melanoma safety override.  When either model flags melanoma above the
 * safety threshold the ensemble takes the MAXIMUM melanoma probability
 * (not the weighted average) to preserve cancer sensitivity.
 *
 * @module ensemble
 */

import type { ClassProbability, LesionClass } from "./types";
import { LESION_LABELS } from "./types";

// ── Constants ───────────────────────────────────────────────────────────

const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const V2_WEIGHT = 0.7;
const V1_WEIGHT = 0.3;
const MEL_SAFETY_THRESHOLD = 0.15;

// ── Public types ────────────────────────────────────────────────────────

export interface EnsembleResult {
	/** Winning class after ensemble weighting */
	topClass: LesionClass;
	/** Confidence (probability) of the winning class */
	confidence: number;
	/** Full probability distribution, sorted descending */
	probabilities: ClassProbability[];
	/** Top class from V1 alone */
	v1TopClass: LesionClass;
	/** Top class from V2 alone */
	v2TopClass: LesionClass;
	/** Whether V1 and V2 agree on the top class */
	modelsAgree: boolean;
	/** Human-readable warning when models disagree, null otherwise */
	disagreementWarning: string | null;
	/** Description of the ensemble method used */
	ensembleMethod: string;
}

// ── Helpers ─────────────────────────────────────────────────────────────

/** Convert a ClassProbability[] to a map keyed by class name. */
function toMap(probs: ClassProbability[]): Map<LesionClass, number> {
	const m = new Map<LesionClass, number>();
	for (const p of probs) {
		m.set(p.className, p.probability);
	}
	return m;
}

/** Return the class with the highest probability from a map. */
function topClassOf(m: Map<LesionClass, number>): LesionClass {
	let best: LesionClass = CLASSES[0];
	let bestP = -1;
	for (const [cls, p] of m) {
		if (p > bestP) {
			bestP = p;
			best = cls;
		}
	}
	return best;
}

// ── Core ensemble function ──────────────────────────────────────────────

/**
 * Combine V1 and V2 model outputs into a single classification using
 * a weighted average with a melanoma safety override.
 *
 * @param v1Probs - Class probabilities from V1 (e.g. Anwarkh1 ViT)
 * @param v2Probs - Class probabilities from V2 (e.g. actavkid ViT)
 * @returns Merged ensemble result
 */
export function ensembleClassify(
	v1Probs: ClassProbability[],
	v2Probs: ClassProbability[],
): EnsembleResult {
	// 1. Build probability maps
	const v1Map = toMap(v1Probs);
	const v2Map = toMap(v2Probs);

	const v1Top = topClassOf(v1Map);
	const v2Top = topClassOf(v2Map);

	// 2. Weighted average: 0.7 * V2 + 0.3 * V1
	const merged = new Map<LesionClass, number>();
	for (const cls of CLASSES) {
		const p1 = v1Map.get(cls) ?? 0;
		const p2 = v2Map.get(cls) ?? 0;
		merged.set(cls, V1_WEIGHT * p1 + V2_WEIGHT * p2);
	}

	// 3. Melanoma safety override: if EITHER model flags mel above the
	//    threshold, take the MAXIMUM mel probability instead of the
	//    weighted average.  Cancer sensitivity over specificity.
	const v1Mel = v1Map.get("mel") ?? 0;
	const v2Mel = v2Map.get("mel") ?? 0;
	const melSafetyTriggered = v1Mel > MEL_SAFETY_THRESHOLD || v2Mel > MEL_SAFETY_THRESHOLD;

	if (melSafetyTriggered) {
		merged.set("mel", Math.max(v1Mel, v2Mel));
	}

	// 4. Normalize to sum = 1
	let sum = 0;
	for (const p of merged.values()) sum += p;
	if (sum > 0) {
		for (const cls of CLASSES) {
			merged.set(cls, (merged.get(cls) ?? 0) / sum);
		}
	}

	// 5. Build sorted probability array
	const probabilities: ClassProbability[] = CLASSES.map((cls) => ({
		className: cls,
		probability: merged.get(cls) ?? 0,
		label: LESION_LABELS[cls],
	})).sort((a, b) => b.probability - a.probability);

	const topClass = probabilities[0].className;
	const confidence = probabilities[0].probability;

	// 6. Detect disagreement & generate warning
	const modelsAgree = v1Top === v2Top;
	const melDisagreement =
		!modelsAgree && (v1Top === "mel" || v2Top === "mel");

	let disagreementWarning: string | null = null;
	if (melDisagreement) {
		disagreementWarning =
			"Models disagree: one suggests melanoma. See a dermatologist to be safe.";
	} else if (!modelsAgree) {
		disagreementWarning =
			`Models disagree: V1 suggests ${LESION_LABELS[v1Top]}, V2 suggests ${LESION_LABELS[v2Top]}. Consider clinical correlation.`;
	}

	return {
		topClass,
		confidence,
		probabilities,
		v1TopClass: v1Top,
		v2TopClass: v2Top,
		modelsAgree,
		disagreementWarning,
		ensembleMethod: melSafetyTriggered
			? `weighted-avg (${V1_WEIGHT}/${V2_WEIGHT}) + mel-safety-override`
			: `weighted-avg (${V1_WEIGHT}/${V2_WEIGHT})`,
	};
}
