/**
 * Meta-Classifier: Fuses neural network output with clinical feature scores
 *
 * The ViT model looks at pixels. The ABCDE/TDS/7-point scores look at
 * clinical features. When they AGREE, confidence goes up. When they
 * DISAGREE (e.g. ViT says melanoma but ABCDE says benign), confidence
 * comes down. This directly reduces false positives by suppressing
 * melanoma probability when clinical features suggest a benign lesion.
 *
 * Thresholds derived from:
 * - Stolz ABCD rule (TDS benign cutoff 4.75)
 * - Argenziano 7-point checklist (biopsy threshold 3)
 * - AAD ABCDE screening criteria
 */

import type { ClassProbability, LesionClass, ABCDEScores } from "./types";

export interface MetaClassification {
	/** Probabilities after clinical feature adjustment, sum to 1 */
	adjustedProbabilities: ClassProbability[];
	/** Top class after adjustment */
	adjustedTopClass: LesionClass;
	/** Confidence of the adjusted top class */
	adjustedConfidence: number;
	/** Raw ViT/neural network melanoma probability */
	neuralConfidence: number;
	/** Clinical suspicion score derived from ABCDE + TDS + 7-point (0-1) */
	clinicalConfidence: number;
	/** Whether neural and clinical signals agree or disagree */
	agreement: "concordant" | "discordant" | "neutral";
	/** Human-readable explanation of the adjustment */
	adjustmentReason: string;
}

/** Thresholds for classifying neural and clinical signals as high/low */
const NEURAL_HIGH = 0.3;
const CLINICAL_HIGH = 0.35;

/**
 * Compute a clinical suspicion score (0-1) from ABCDE, TDS, and 7-point data.
 *
 * Each positive clinical indicator adds to the score. The maximum possible
 * is 1.0, which represents a lesion with every suspicious feature present.
 */
function computeClinicalSuspicion(
	abcde: ABCDEScores | null,
	tdsScore: number | null,
	sevenPointScore: number | null,
	diameterMm: number | null,
): number {
	let score = 0;

	if (abcde) {
		// Asymmetry >= 1 axis: suspicious
		if (abcde.asymmetry >= 1) score += 0.15;

		// Border irregularity in >= 4 of 8 segments
		if (abcde.border >= 4) score += 0.10;

		// 3 or more distinct colors
		if (abcde.color >= 3) score += 0.10;

		// Blue-white veil detected (check colorsDetected array)
		const hasBlueWhiteVeil = abcde.colorsDetected.some(
			(c) => c === "blue-gray" || c === "blue-white",
		);
		if (hasBlueWhiteVeil) score += 0.10;
	}

	// Diameter >= 6mm (use explicit param or fall back to ABCDE)
	const effectiveDiameter = diameterMm ?? abcde?.diameterMm ?? 0;
	if (effectiveDiameter >= 6) score += 0.15;

	// TDS > 4.75 (Stolz benign cutoff)
	if (tdsScore !== null && tdsScore > 4.75) score += 0.20;

	// 7-point checklist >= 3 (biopsy threshold)
	if (sevenPointScore !== null && sevenPointScore >= 3) score += 0.20;

	return Math.min(score, 1.0);
}

/**
 * Determine agreement between neural and clinical signals.
 *
 * - concordant-high: both above threshold (reinforce melanoma)
 * - concordant-low: both below threshold (suppress melanoma)
 * - discordant-neural-high: neural flags melanoma, clinical does not
 * - discordant-clinical-high: clinical features suspicious, neural is not
 * - neutral: one or both signals are ambiguous
 */
function classifyAgreement(
	neuralMelProb: number,
	clinicalScore: number,
): {
	agreement: "concordant" | "discordant" | "neutral";
	multiplier: number;
	reason: string;
} {
	const neuralHigh = neuralMelProb >= NEURAL_HIGH;
	const neuralLow = neuralMelProb < NEURAL_HIGH;
	const clinicalHigh = clinicalScore >= CLINICAL_HIGH;
	const clinicalLow = clinicalScore < CLINICAL_HIGH;

	// Both agree it looks suspicious
	if (neuralHigh && clinicalHigh) {
		return {
			agreement: "concordant",
			multiplier: 1.2,
			reason: "Clinical features support the AI classification",
		};
	}

	// Both agree it looks benign
	if (neuralLow && clinicalLow) {
		return {
			agreement: "concordant",
			multiplier: 0.8,
			reason: "Clinical features and AI both suggest low risk",
		};
	}

	// Neural says melanoma, clinical says benign -- most important case
	// for reducing false positives. A symmetric, single-color, small lesion
	// with low TDS is unlikely melanoma even if the ViT flags it.
	if (neuralHigh && clinicalLow) {
		return {
			agreement: "discordant",
			multiplier: 0.6,
			reason:
				"Clinical features (low ABCDE score) suggest this may be benign despite AI flag",
		};
	}

	// Clinical says suspicious, neural disagrees -- safety-critical,
	// boost melanoma probability because clinical features are warning signs
	if (neuralLow && clinicalHigh) {
		return {
			agreement: "discordant",
			multiplier: 1.3,
			reason:
				"Clinical features (high TDS/7-point) suggest increased risk despite moderate AI confidence",
		};
	}

	// Fallback: ambiguous / borderline case
	return {
		agreement: "neutral",
		multiplier: 1.0,
		reason: "Clinical features are inconclusive; relying on AI classification",
	};
}

/**
 * Renormalize an array of probabilities so they sum to 1.
 * Preserves relative ordering.
 */
function renormalize(probs: ClassProbability[]): ClassProbability[] {
	const total = probs.reduce((sum, p) => sum + p.probability, 0);
	if (total <= 0) return probs;

	return probs.map((p) => ({
		...p,
		probability: p.probability / total,
	}));
}

/**
 * Combine neural network output with clinical feature scores to produce
 * an adjusted classification. This is the core PPV-improvement mechanism:
 * when the ViT says "melanoma" but ABCDE/TDS/7-point say "benign",
 * the melanoma probability is pulled down, reducing false positives.
 *
 * @param neuralProbs - Raw probability distribution from the ViT/neural network
 * @param abcdeScores - ABCDE scoring results (null if not yet computed)
 * @param tdsScore - Total Dermoscopy Score (null if unavailable)
 * @param sevenPointScore - 7-point checklist score (null if unavailable)
 * @param diameterMm - Measured lesion diameter in mm (null if unavailable)
 * @returns Meta-classification with adjusted probabilities and agreement info
 */
export function metaClassify(
	neuralProbs: ClassProbability[],
	abcdeScores: ABCDEScores | null,
	tdsScore: number | null,
	sevenPointScore: number | null,
	diameterMm: number | null,
): MetaClassification {
	// Find melanoma probability from the neural network output
	const melProb = neuralProbs.find((p) => p.className === "mel");
	const neuralMelConfidence = melProb?.probability ?? 0;

	// If we have no clinical data at all, return neural output unchanged
	if (!abcdeScores && tdsScore === null && sevenPointScore === null) {
		const sorted = [...neuralProbs].sort(
			(a, b) => b.probability - a.probability,
		);
		return {
			adjustedProbabilities: sorted,
			adjustedTopClass: sorted[0].className,
			adjustedConfidence: sorted[0].probability,
			neuralConfidence: neuralMelConfidence,
			clinicalConfidence: 0,
			agreement: "neutral",
			adjustmentReason:
				"No clinical scoring data available; using AI classification only",
		};
	}

	// Compute clinical suspicion
	const clinicalScore = computeClinicalSuspicion(
		abcdeScores,
		tdsScore,
		sevenPointScore,
		diameterMm,
	);

	// Determine agreement and get the adjustment multiplier
	const { agreement, multiplier, reason } = classifyAgreement(
		neuralMelConfidence,
		clinicalScore,
	);

	// Apply multiplier to melanoma probability
	const adjusted = neuralProbs.map((p) => {
		if (p.className === "mel") {
			return {
				...p,
				probability: Math.max(0, p.probability * multiplier),
			};
		}
		return { ...p };
	});

	// Renormalize so probabilities sum to 1
	const normalized = renormalize(adjusted);

	// Sort descending and determine new top class
	const sorted = [...normalized].sort(
		(a, b) => b.probability - a.probability,
	);

	return {
		adjustedProbabilities: sorted,
		adjustedTopClass: sorted[0].className,
		adjustedConfidence: sorted[0].probability,
		neuralConfidence: neuralMelConfidence,
		clinicalConfidence: clinicalScore,
		agreement,
		adjustmentReason: reason,
	};
}
