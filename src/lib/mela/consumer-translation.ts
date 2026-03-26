/**
 * Consumer Translation Module
 *
 * Translates pattern analysis results into plain English
 * that a regular person can understand.
 *
 * Uses Bayesian pattern stratification (risk-stratification.ts) to compute
 * post-test probability from the model's continuous output and real-world
 * prevalence, replacing the old binary classification that suffered from
 * a PPV of only 8.9% at real-world 2% melanoma prevalence.
 *
 * The goal: someone with zero medical training should be able to
 * read the output and understand the analysis results.
 */

import { assessRisk, type BayesianRiskLevel, type RiskAssessment } from "./risk-stratification";

export type ConsumerRiskLevel = "green" | "yellow" | "orange" | "red";

export interface ConsumerResult {
	headline: string;
	riskLevel: ConsumerRiskLevel;
	riskColor: string;
	explanation: string;
	action: string;
	shouldSeeDoctor: boolean;
	urgency: "none" | "routine" | "soon" | "urgent";
	medicalTerm: string;
	confidence: number;
	/** Bayesian risk assessment (present when full probabilities are available) */
	riskAssessment?: RiskAssessment;
}

const TRANSLATIONS: Record<
	string,
	{
		name: string;
		risk: ConsumerRiskLevel;
		message: string;
		action: string;
		explanation: string;
		shouldSeeDoctor: boolean;
		urgency: "none" | "routine" | "soon" | "urgent";
	}
> = {
	nv: {
		name: "Common mole",
		risk: "green",
		message: "This looks like a typical mole -- low concern.",
		action: "Monitor for changes. Re-check in 6 months.",
		explanation:
			"This appears to be a melanocytic nevus (a regular mole). Most moles are harmless. Watch for changes in size, shape, or color over time.",
		shouldSeeDoctor: false,
		urgency: "none",
	},
	bkl: {
		name: "Age spot / seborrheic keratosis",
		risk: "green",
		message: "This looks like a common age spot -- typically harmless.",
		action: "No action needed unless it bothers you cosmetically.",
		explanation:
			"This appears to be a benign keratosis, commonly known as an age spot or liver spot. These are patterns typically considered benign in medical literature.",
		shouldSeeDoctor: false,
		urgency: "none",
	},
	df: {
		name: "Firm skin bump",
		risk: "green",
		message: "This looks like a dermatofibroma -- usually benign.",
		action: "No treatment needed unless it's painful or growing.",
		explanation:
			"Dermatofibromas are firm bumps under the skin that are almost always harmless. They're caused by a buildup of collagen.",
		shouldSeeDoctor: false,
		urgency: "none",
	},
	vasc: {
		name: "Blood vessel spot",
		risk: "green",
		message:
			"This appears to be a vascular lesion -- typically harmless.",
		action: "Consider consulting a healthcare provider if it bleeds frequently or grows rapidly.",
		explanation:
			"Vascular lesions are caused by blood vessels near the skin surface. Cherry angiomas and spider angiomas are common and benign.",
		shouldSeeDoctor: false,
		urgency: "none",
	},
	akiec: {
		name: "Sun damage spot",
		risk: "yellow",
		message: "This may be sun damage that should be monitored.",
		action: "Consider having a healthcare provider evaluate this at your next visit.",
		explanation:
			"This may be an actinic keratosis -- a rough, scaly patch caused by years of sun exposure. In medical literature, while not concerning on its own, some can develop into more serious conditions if untreated. Easy to treat when caught early.",
		shouldSeeDoctor: true,
		urgency: "routine",
	},
	bcc: {
		name: "Needs evaluation",
		risk: "orange",
		message:
			"This has features that should be evaluated by a doctor.",
		action: "Professional evaluation may be helpful. Consider scheduling an appointment.",
		explanation:
			"The analysis found visual similarity to reference patterns for basal cell carcinoma -- described in medical literature as the most common and most treatable form of skin concern. It almost never spreads to other parts of the body. Early treatment is straightforward and highly effective.",
		shouldSeeDoctor: true,
		urgency: "soon",
	},
	mel: {
		name: "Consider professional evaluation",
		risk: "red",
		message:
			"This has features that may warrant professional evaluation.",
		action: "This pattern may warrant professional evaluation.",
		explanation:
			"The analysis found visual similarity to reference patterns associated with melanoma in medical literature -- described as the most serious form of skin concern. This does NOT mean you have a medical condition. But these features may warrant evaluation by a healthcare provider for professional assessment. In medical literature, when caught early, conditions associated with these patterns are highly treatable.",
		shouldSeeDoctor: true,
		urgency: "urgent",
	},
};

const RISK_COLORS: Record<ConsumerRiskLevel, string> = {
	green: "#10b981",
	yellow: "#f59e0b",
	orange: "#f97316",
	red: "#ef4444",
};

/** Map Bayesian risk levels to the existing ConsumerRiskLevel color scheme */
const BAYESIAN_TO_CONSUMER_RISK: Record<BayesianRiskLevel, ConsumerRiskLevel> = {
	"very-high": "red",
	"high": "orange",
	"moderate": "yellow",
	"low": "green",
	"minimal": "green",
};

/** Map Bayesian risk levels to ConsumerResult urgency values */
const BAYESIAN_TO_URGENCY: Record<BayesianRiskLevel, "none" | "routine" | "soon" | "urgent"> = {
	"very-high": "urgent",
	"high": "soon",
	"moderate": "routine",
	"low": "none",
	"minimal": "none",
};

/**
 * Translate a classification result into consumer-friendly language.
 *
 * When full probability data is available, uses Bayesian risk stratification
 * to compute an honest post-test probability that accounts for real-world
 * prevalence (2% melanoma, 5% BCC, 3% actinic keratosis). This replaces the
 * old binary approach where "melanoma detected" had a PPV of only 8.9%.
 *
 * @param topClass - The top predicted HAM10000 class (e.g. "mel", "nv")
 * @param confidence - Confidence score from the model [0, 1]
 * @param allProbabilities - Optional full probability distribution for Bayesian update
 * @param demographics - Optional age/location for prevalence adjustment
 */
export function translateForConsumer(
	topClass: string,
	confidence: number,
	allProbabilities?: Array<{ className: string; probability: number }>,
	demographics?: { age?: number; bodyLocation?: string },
): ConsumerResult {
	// SAFETY: If the top probability is below 0.40 for ANY class, the model
	// is not confident enough to classify. This prevents spurious "see a
	// dermatologist" results when the image doesn't contain a clear lesion
	// or is ambiguous. The threshold is intentionally conservative: a model
	// that can't even reach 40% for its top pick is essentially guessing.
	if (confidence < 0.40) {
		return {
			headline: "Consider professional evaluation",
			riskLevel: "orange",
			riskColor: RISK_COLORS["orange"],
			explanation:
				"Mela could not analyze this spot with enough confidence. " +
				"This can happen with image quality issues, but it can also happen with " +
				"unusual or flesh-colored features that are difficult for any AI to assess. " +
				"Some pattern types, including amelanotic variants, have very low visual contrast.",
			action:
				"Professional evaluation may be helpful. When the analysis is uncertain, " +
				"consulting a healthcare provider is the safest path. Try retaking the photo " +
				"with better lighting and closer framing, but do not rely solely on a re-scan.",
			shouldSeeDoctor: true,
			urgency: "routine",
			medicalTerm: topClass,
			confidence,
		};
	}

	const translation = TRANSLATIONS[topClass] || TRANSLATIONS["nv"];

	// When we have full probability data, use Bayesian risk stratification
	if (allProbabilities && allProbabilities.length > 0) {
		// Convert array format to Record for assessRisk
		const probRecord: Record<string, number> = {};
		for (const p of allProbabilities) {
			probRecord[p.className] = p.probability;
		}

		const risk = assessRisk(topClass, confidence, probRecord, demographics);

		const effectiveRisk = BAYESIAN_TO_CONSUMER_RISK[risk.riskLevel];
		const effectiveUrgency = BAYESIAN_TO_URGENCY[risk.riskLevel];

		// Use Bayesian headline/action, but keep the medical explanation
		let effectiveAction = risk.action;
		if (confidence < 0.4) {
			effectiveAction =
				"The image quality or classification confidence is low. Consider retaking the photo with better lighting, or consulting a healthcare provider for professional evaluation.";
		}

		return {
			headline: risk.headline,
			riskLevel: effectiveRisk,
			riskColor: risk.color,
			explanation: translation.explanation,
			action: effectiveAction,
			shouldSeeDoctor: risk.riskLevel === "very-high" || risk.riskLevel === "high" || risk.riskLevel === "moderate",
			urgency: effectiveUrgency,
			medicalTerm: topClass,
			confidence,
			riskAssessment: risk,
		};
	}

	// Fallback: no probability data, use static translation
	let effectiveAction = translation.action;
	if (confidence < 0.4) {
		effectiveAction =
			"The image quality or classification confidence is low. Consider retaking the photo with better lighting, or consulting a healthcare provider for professional evaluation.";
	}

	return {
		headline: translation.name,
		riskLevel: translation.risk,
		riskColor: RISK_COLORS[translation.risk],
		explanation: translation.explanation,
		action: effectiveAction,
		shouldSeeDoctor: translation.shouldSeeDoctor,
		urgency: translation.urgency,
		medicalTerm: topClass,
		confidence,
	};
}
