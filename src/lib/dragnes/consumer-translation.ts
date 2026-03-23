/**
 * Consumer Translation Module
 *
 * Translates medical classification results into plain English
 * that a regular person can understand and act on.
 *
 * The goal: someone with zero medical training should be able to
 * read the output and know exactly what to do next.
 */

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

/**
 * Translate a classification result into consumer-friendly language.
 *
 * @param topClass - The top predicted HAM10000 class (e.g. "mel", "nv")
 * @param confidence - Confidence score from the model [0, 1]
 * @param allProbabilities - Optional full probability distribution for cancer-risk aggregation
 */
export function translateForConsumer(
	topClass: string,
	confidence: number,
	allProbabilities?: Array<{ className: string; probability: number }>,
): ConsumerResult {
	const translation = TRANSLATIONS[topClass] || TRANSLATIONS["nv"];

	// Start with the static translation values
	let effectiveRisk = translation.risk;
	let effectiveAction = translation.action;

	// If confidence is low, add a caveat to the action
	if (confidence < 0.4) {
		effectiveAction =
			"The image quality or classification confidence is low. Consider retaking the photo with better lighting, or see a dermatologist for a definitive assessment.";
	}

	// If ANY cancer class has > 30% combined probability, upgrade to at least yellow
	if (allProbabilities) {
		const cancerProb = allProbabilities
			.filter((p) =>
				["mel", "bcc", "akiec"].includes(p.className),
			)
			.reduce((sum, p) => sum + p.probability, 0);

		if (cancerProb > 0.3 && effectiveRisk === "green") {
			effectiveRisk = "yellow";
			effectiveAction =
				"Some concerning features detected. Have a dermatologist take a look at your next visit.";
		}
	}

	return {
		headline: translation.name,
		riskLevel: effectiveRisk,
		riskColor: RISK_COLORS[effectiveRisk],
		explanation: translation.explanation,
		action: effectiveAction,
		shouldSeeDoctor: translation.shouldSeeDoctor,
		urgency: translation.urgency,
		medicalTerm: topClass,
		confidence,
	};
}
