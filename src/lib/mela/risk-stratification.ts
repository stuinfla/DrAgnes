/**
 * Bayesian Risk Stratification Module
 *
 * Fixes the PPV problem: binary classification gives PPV of 47.6% at test
 * prevalence and 8.9% at real-world 2% prevalence. Instead of binary yes/no,
 * we compute a post-test probability using Bayes' theorem with the model's
 * continuous probability output and real-world prevalence.
 *
 * Every result becomes actionable and honest about uncertainty.
 */

export type BayesianRiskLevel = "very-high" | "high" | "moderate" | "low" | "minimal";

export interface RiskAssessment {
	/** Bayesian risk tier */
	riskLevel: BayesianRiskLevel;
	/** Post-test probability after Bayesian update [0, 1] */
	postTestProbability: number;
	/** Pre-test probability (prevalence prior, possibly age-adjusted) */
	preTestProbability: number;
	/** Likelihood ratio derived from model confidence */
	likelihoodRatio: number;
	/** Consumer-facing headline */
	headline: string;
	/** What to do next */
	action: string;
	/** Timeframe for action */
	urgency: string;
	/** UI color for the risk tier */
	color: string;
}

// ---------------------------------------------------------------------------
// Epidemiological base prevalence rates
// (proportion of lesions seen in screening that turn out to be each type)
// ---------------------------------------------------------------------------
const MELANOMA_PREVALENCE = 0.02;
const BCC_PREVALENCE = 0.05;
const AKIEC_PREVALENCE = 0.03;

/** Combined malignant/pre-malignant base prevalence */
const MALIGNANT_PREVALENCE = MELANOMA_PREVALENCE + BCC_PREVALENCE + AKIEC_PREVALENCE; // 0.10

// ---------------------------------------------------------------------------
// Age-based prevalence multipliers (melanoma incidence scales with age)
// ---------------------------------------------------------------------------
const AGE_MULTIPLIERS: Array<{ maxAge: number; multiplier: number }> = [
	{ maxAge: 29, multiplier: 0.3 },
	{ maxAge: 50, multiplier: 0.7 },
	{ maxAge: 70, multiplier: 1.5 },
	{ maxAge: Infinity, multiplier: 2.5 },
];

function getAgeMultiplier(age: number | undefined): number {
	if (age === undefined || age < 0) return 1.0;
	for (const bracket of AGE_MULTIPLIERS) {
		if (age <= bracket.maxAge) return bracket.multiplier;
	}
	return 1.0;
}

// ---------------------------------------------------------------------------
// Clinical history risk multipliers (Dr. Chang feedback, ADR-130)
// ---------------------------------------------------------------------------
export interface ClinicalHistory {
	/** Is this lesion new or longstanding? */
	isNew?: "new" | "months" | "years" | "unsure";
	/** Has it changed recently? */
	hasChanged?: "yes" | "no" | "unsure";
	/** Has it been biopsied before? */
	previouslyBiopsied?: "yes" | "no";
	/** Family history of melanoma? */
	familyHistoryMelanoma?: "yes" | "no" | "unsure";
	/** Current symptoms? */
	symptoms?: ("itching" | "bleeding" | "pain" | "none")[];
}

/**
 * Compute a combined risk multiplier from clinical history.
 *
 * These multipliers are applied to the pre-test prevalence before
 * the Bayesian update. A new, changing mole with family history in
 * someone with symptoms has a substantially higher pre-test probability
 * than a stable, asymptomatic mole.
 *
 * Evidence: Gandini et al. 2005 (family history RR 1.74), Abbasi et al.
 * 2004 (ABCDE evolution), Grob & Bonerandi 1998 (change as predictor).
 */
function getClinicalHistoryMultiplier(history: ClinicalHistory | undefined): number {
	if (!history) return 1.0;

	let multiplier = 1.0;

	// New lesion: higher risk than longstanding
	if (history.isNew === "new") multiplier *= 1.3;
	else if (history.isNew === "months") multiplier *= 1.15;
	// "years" and "unsure" = no adjustment

	// Recent change: strong predictor (the "E" in ABCDE)
	if (history.hasChanged === "yes") multiplier *= 1.5;

	// Previously biopsied: could be recurrent nevi (looks like cancer but isn't)
	// Lower the multiplier slightly to account for this
	if (history.previouslyBiopsied === "yes") multiplier *= 0.8;

	// Family history: RR ~1.74 for first-degree relative (Gandini 2005)
	if (history.familyHistoryMelanoma === "yes") multiplier *= 1.7;

	// Symptoms: itching/bleeding/pain are concerning
	if (history.symptoms && history.symptoms.length > 0) {
		const hasSymptoms = history.symptoms.some((s) => s !== "none");
		if (hasSymptoms) multiplier *= 1.3;
		if (history.symptoms.includes("bleeding")) multiplier *= 1.2; // extra for bleeding
	}

	return multiplier;
}

// ---------------------------------------------------------------------------
// Risk-level thresholds and messaging
// ---------------------------------------------------------------------------
interface RiskTier {
	level: BayesianRiskLevel;
	minProbability: number;
	headline: string;
	action: string;
	urgency: string;
	color: string;
}

const RISK_TIERS: RiskTier[] = [
	{
		level: "very-high",
		minProbability: 0.50,
		headline: "Elevated patterns detected -- professional evaluation may be appropriate",
		action: "Consider scheduling a professional evaluation soon",
		urgency: "consider soon",
		color: "#ef4444", // red
	},
	{
		level: "high",
		minProbability: 0.20,
		headline: "Consider professional evaluation",
		action: "Consider scheduling a professional evaluation",
		urgency: "consider evaluation",
		color: "#f97316", // orange
	},
	{
		level: "moderate",
		minProbability: 0.05,
		headline: "Worth monitoring",
		action: "Photograph monthly, consider professional evaluation if changes occur",
		urgency: "monitor monthly",
		color: "#f59e0b", // yellow
	},
	{
		level: "low",
		minProbability: 0.01,
		headline: "Low pattern concern",
		action: "Continue routine skin awareness checks",
		urgency: "routine schedule",
		color: "#10b981", // green
	},
	{
		level: "minimal",
		minProbability: 0,
		headline: "No concerning patterns detected",
		action: "Normal skin awareness schedule",
		urgency: "normal schedule",
		color: "#14b8a6", // teal
	},
];

function getTier(postTestProbability: number): RiskTier {
	for (const tier of RISK_TIERS) {
		if (postTestProbability >= tier.minProbability) return tier;
	}
	// Fallback (should never reach here because minimal has minProbability 0)
	return RISK_TIERS[RISK_TIERS.length - 1];
}

// ---------------------------------------------------------------------------
// Core Bayesian risk assessment
// ---------------------------------------------------------------------------

/**
 * Assess risk using Bayesian post-test probability.
 *
 * Instead of reporting the model's raw confidence (which conflates sensitivity
 * with PPV), we compute:
 *
 *   1. Aggregate the model's malignant-class probabilities (mel + bcc + akiec).
 *   2. Convert that into a likelihood ratio.
 *   3. Multiply by the age-adjusted pre-test odds (prevalence).
 *   4. Convert back to a post-test probability.
 *
 * The result is an honest statement about how likely the lesion actually is
 * to be malignant given the model output AND the base rate.
 *
 * @param topClass - The model's top predicted HAM10000 class
 * @param modelConfidence - The model's probability for the top class [0, 1]
 * @param allProbabilities - Full probability distribution keyed by class name
 * @param demographics - Optional age and body location for prior adjustment
 */
export function assessRisk(
	topClass: string,
	modelConfidence: number,
	allProbabilities: Record<string, number>,
	demographics?: { age?: number; bodyLocation?: string },
	clinicalHistory?: ClinicalHistory,
): RiskAssessment {
	// 1. Aggregate malignant probability from the model
	const melProb = allProbabilities["mel"] ?? 0;
	const bccProb = allProbabilities["bcc"] ?? 0;
	const akiecProb = allProbabilities["akiec"] ?? 0;
	const malignantProb = Math.min(melProb + bccProb + akiecProb, 0.9999);

	// 2. Compute age-adjusted and clinical-history-adjusted prevalence
	const ageMultiplier = getAgeMultiplier(demographics?.age);
	const historyMultiplier = getClinicalHistoryMultiplier(clinicalHistory);
	const adjustedPrevalence = Math.min(MALIGNANT_PREVALENCE * ageMultiplier * historyMultiplier, 0.5);

	// 3. Bayesian update
	//    Likelihood ratio: LR = P(model output | disease) / P(model output | no disease)
	//    Simplified: treat the malignant probability as a continuous test result
	const clampedMalignant = Math.max(malignantProb, 0.0001);
	const clampedBenign = Math.max(1 - malignantProb, 0.0001);
	const likelihoodRatio = clampedMalignant / clampedBenign;

	//    Pre-test odds = prevalence / (1 - prevalence)
	const preTestOdds = adjustedPrevalence / (1 - adjustedPrevalence);

	//    Post-test odds = pre-test odds * LR
	const postTestOdds = preTestOdds * likelihoodRatio;

	//    Post-test probability = post-test odds / (1 + post-test odds)
	const postTestProbability = postTestOdds / (1 + postTestOdds);

	// 4. Map to risk tier
	const tier = getTier(postTestProbability);

	return {
		riskLevel: tier.level,
		postTestProbability,
		preTestProbability: adjustedPrevalence,
		likelihoodRatio,
		headline: tier.headline,
		action: tier.action,
		urgency: tier.urgency,
		color: tier.color,
	};
}
