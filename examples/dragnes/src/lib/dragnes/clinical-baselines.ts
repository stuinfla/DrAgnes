/**
 * Clinical baselines calibrated against DermaSensor (FDA DEN230008),
 * HAM10000 dataset, AAD 2024-2026 guidelines, and dermoscopy literature.
 *
 * Sources:
 * - DERM-SUCCESS FDA pivotal study (1,579 lesions)
 * - DERM-ASSESS III multicenter melanoma study (440 lesions)
 * - Dermoscopedia ABCD rule and 7-point checklist
 * - AAD screening guidelines 2024-2026
 */

/** DermaSensor benchmark performance (FDA DEN230008, cleared Jan 2024) */
export const DERMASENSOR_BENCHMARKS = {
	sensitivity: { melanoma: 0.902, bcc: 0.978, scc: 0.977, overall: 0.955 },
	specificity: { overall: 0.207, dermatologySetting: 0.325 },
	npv: 0.966,
	ppv: 0.160,
	auroc: 0.758,
	falseNegativeRate: { melanoma: 0.045 },
	fitzpatrick: {
		fst_1_3: { sensitivity: 0.96, auroc: 0.779 },
		fst_4_6: { sensitivity: 0.92, auroc: 0.764 },
		maxDisparity: 0.04,
	},
} as const;

/** DrAgnes classification targets */
export const DRAGNES_TARGETS = {
	sensitivity: { melanoma: 0.95, bcc: 0.95, scc: 0.92, overall: 0.95 },
	specificity: { minimum: 0.35, target: 0.50 },
	falseNegativeRate: { melanomaCeiling: 0.05 },
	npv: { minimum: 0.96 },
	fitzpatrickDisparity: { maxGap: 0.05 },
} as const;

/** Clinical decision thresholds */
export const DECISION_THRESHOLDS = {
	/** P(malignant) = sum of mel + bcc + akiec probabilities */
	biopsyRecommend: 0.30,
	urgentReferral: 0.50,
	monitoring: { min: 0.10, max: 0.30 },
	reassurance: 0.10,
} as const;

/** Total Dermoscopy Score (TDS) - weighted ABCD formula */
export const TDS = {
	weights: { A: 1.3, B: 0.1, C: 0.5, D: 0.5 },
	/** TDS < 4.75: benign (90.3% of nevi fall here) */
	benignCutoff: 4.75,
	/** TDS 4.75-5.45: suspicious, close monitoring or biopsy */
	suspiciousCutoff: 5.45,
	/** TDS > 5.45: probable malignant, biopsy recommended */
	malignantCutoff: 5.45,
} as const;

/** Compute Total Dermoscopy Score from ABCDE components */
export function computeTDS(asymmetry: number, border: number, color: number, structures: number): number {
	return (
		asymmetry * TDS.weights.A +
		border * TDS.weights.B +
		color * TDS.weights.C +
		structures * TDS.weights.D
	);
}

/** Get risk level from TDS score */
export function tdsRiskLevel(tds: number): "low" | "moderate" | "high" | "critical" {
	if (tds < TDS.benignCutoff) return "low";
	if (tds <= TDS.suspiciousCutoff) return "moderate";
	if (tds <= 7.0) return "high";
	return "critical";
}

/** 7-Point Dermoscopy Checklist */
export const SEVEN_POINT_CHECKLIST = {
	majorCriteria: {
		atypicalNetwork: 2,
		blueWhiteVeil: 2,
		atypicalVascular: 2,
	},
	minorCriteria: {
		irregularStreaks: 1,
		irregularDots: 1,
		irregularBlotches: 1,
		regressionStructures: 1,
	},
	biopsyThreshold: 3,
	sensitivity: 0.95,
	specificity: 0.75,
} as const;

/** Compute 7-point checklist score from detected structures */
export function computeSevenPointScore(structures: {
	hasIrregularNetwork: boolean;
	hasBlueWhiteVeil: boolean;
	hasStreaks: boolean;
	hasIrregularGlobules: boolean;
	hasRegressionStructures: boolean;
}): { score: number; recommendation: string; details: string[] } {
	const details: string[] = [];
	let score = 0;

	if (structures.hasIrregularNetwork) {
		score += SEVEN_POINT_CHECKLIST.majorCriteria.atypicalNetwork;
		details.push("Atypical pigment network (+2)");
	}
	if (structures.hasBlueWhiteVeil) {
		score += SEVEN_POINT_CHECKLIST.majorCriteria.blueWhiteVeil;
		details.push("Blue-white veil (+2)");
	}
	if (structures.hasStreaks) {
		score += SEVEN_POINT_CHECKLIST.minorCriteria.irregularStreaks;
		details.push("Irregular streaks (+1)");
	}
	if (structures.hasIrregularGlobules) {
		score += SEVEN_POINT_CHECKLIST.minorCriteria.irregularDots;
		details.push("Irregular dots/globules (+1)");
	}
	if (structures.hasRegressionStructures) {
		score += SEVEN_POINT_CHECKLIST.minorCriteria.regressionStructures;
		details.push("Regression structures (+1)");
	}

	const recommendation = score >= SEVEN_POINT_CHECKLIST.biopsyThreshold
		? "Biopsy recommended (7-point score >= 3)"
		: score >= 2
			? "Close monitoring recommended"
			: "No immediate concern from 7-point checklist";

	return { score, recommendation, details };
}

/** Confidence-stratified action mapping (derived from DermaSensor spectral score PPV) */
export const CONFIDENCE_STRATIFICATION = [
	{ range: [0.00, 0.10] as const, ppv: 0.03, action: "reassurance" as const, label: "Low concern" },
	{ range: [0.10, 0.30] as const, ppv: 0.06, action: "monitor_3mo" as const, label: "Monitor in 3 months" },
	{ range: [0.30, 0.60] as const, ppv: 0.18, action: "biopsy_consider" as const, label: "Consider biopsy" },
	{ range: [0.60, 0.80] as const, ppv: 0.40, action: "biopsy_recommend" as const, label: "Biopsy recommended" },
	{ range: [0.80, 1.00] as const, ppv: 0.61, action: "urgent_referral" as const, label: "Urgent dermatology referral" },
] as const;

/** Get action recommendation from malignant probability */
export function getConfidenceAction(malignantProbability: number): typeof CONFIDENCE_STRATIFICATION[number] {
	for (const level of CONFIDENCE_STRATIFICATION) {
		if (malignantProbability >= level.range[0] && malignantProbability < level.range[1]) {
			return level;
		}
	}
	return CONFIDENCE_STRATIFICATION[CONFIDENCE_STRATIFICATION.length - 1];
}
