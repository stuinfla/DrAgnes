/**
 * Consumer Translation Module
 *
 * Translates medical classification results into plain English
 * that a regular person can understand and act on.
 *
 * Uses Bayesian risk stratification (risk-stratification.ts) to compute
 * post-test probability from the model's continuous output and real-world
 * prevalence, replacing the old binary classification that suffered from
 * a PPV of only 8.9% at real-world 2% melanoma prevalence.
 *
 * The goal: someone with zero medical training should be able to
 * read the output and know exactly what to do next.
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
			"This appears to be a benign keratosis, commonly known as an age spot or liver spot. These are very common and not cancerous.",
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
		action: "See a doctor if it bleeds frequently or grows rapidly.",
		explanation:
			"Vascular lesions are caused by blood vessels near the skin surface. Cherry angiomas and spider angiomas are common and benign.",
		shouldSeeDoctor: false,
		urgency: "none",
	},
	akiec: {
		name: "Sun damage spot",
		risk: "yellow",
		message: "This may be sun damage that should be monitored.",
		action: "Have a dermatologist check this at your next visit.",
		explanation:
			"This may be an actinic keratosis -- a rough, scaly patch caused by years of sun exposure. While not cancer itself, some can develop into squamous cell carcinoma if untreated. Easy to treat when caught early.",
		shouldSeeDoctor: true,
		urgency: "routine",
	},
	bcc: {
		name: "Needs evaluation",
		risk: "orange",
		message:
			"This has features that should be evaluated by a doctor.",
		action: "Schedule a dermatology appointment within 1 month.",
		explanation:
			"The AI detected patterns consistent with basal cell carcinoma -- the most common and most treatable form of skin cancer. It almost never spreads to other parts of the body. Early treatment is straightforward and highly effective.",
		shouldSeeDoctor: true,
		urgency: "soon",
	},
	mel: {
		name: "See a dermatologist",
		risk: "red",
		message:
			"This has features that need professional evaluation promptly.",
		action: "Please see a dermatologist within 2 weeks.",
		explanation:
			"The AI detected patterns associated with melanoma -- the most serious form of skin cancer. This does NOT mean you have cancer. But these features warrant prompt evaluation by a dermatologist. When caught early, melanoma is highly treatable with a 99% survival rate.",
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
				"The image quality or classification confidence is low. Consider retaking the photo with better lighting, or see a dermatologist for a definitive assessment.";
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
			"The image quality or classification confidence is low. Consider retaking the photo with better lighting, or see a dermatologist for a definitive assessment.";
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
