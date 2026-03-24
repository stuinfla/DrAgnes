/**
 * DrAgnes Anonymization Pipeline (ADR-126 Phase 2)
 *
 * Transforms raw classification outputs and patient demographics into
 * de-identified, differentially private AnonymizedCase records suitable
 * for sharing with pi-brain collective intelligence.
 *
 * Privacy guarantees:
 *   - Laplace differential privacy noise on all probability values (eps=1.0)
 *   - Age reduced to decade bucket ("50s") -- never exact age
 *   - Timestamps stripped to ISO date only (no hours/minutes/seconds)
 *   - No raw images, device identifiers, or patient identifiers included
 */

/** De-identified case record ready for collective sharing */
export interface AnonymizedCase {
	/** Classification output (noised) */
	probabilities: Record<string, number>;
	topClass: string;
	confidence: number;

	/** Clinical context (de-identified) */
	bodyLocation: string;
	ageDecade: string | null;
	sex: string | null;

	/** Measurement */
	diameterMm: number | null;
	abcdeScore: number | null;

	/** Outcome (when available) */
	outcome:
		| "concordant"
		| "discordant_overcall"
		| "discordant_missed"
		| "biopsied"
		| null;
	pathologyResult: string | null;

	/** Metadata */
	timestamp: string;
	modelVersion: string;
}

/** Default differential privacy epsilon -- moderate privacy/utility tradeoff */
const DEFAULT_EPSILON = 1.0;

/**
 * Sample from a Laplace distribution centered at 0 with the given scale.
 * Uses the inverse CDF method: X = -b * sign(U) * ln(1 - 2|U|)
 * where U ~ Uniform(-0.5, 0.5).
 */
function laplaceSample(scale: number): number {
	const u = Math.random() - 0.5;
	return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}

/**
 * Convert an exact age to a decade bucket string.
 * Returns null for undefined/invalid ages.
 *
 * @example ageToDecade(53) => "50s"
 * @example ageToDecade(7) => "0s"
 */
function ageToDecade(age: number | undefined): string | null {
	if (age === undefined || age < 0 || !Number.isFinite(age)) return null;
	const decade = Math.floor(age / 10) * 10;
	return `${decade}s`;
}

/**
 * Strip an ISO datetime string to date-only (YYYY-MM-DD) for k-anonymity.
 * Removing time component prevents temporal re-identification.
 */
function stripToDateOnly(iso: string): string {
	return iso.slice(0, 10);
}

/**
 * Anonymize a classification result with differential privacy noise
 * and demographic reduction for safe collective sharing.
 *
 * Pipeline:
 *   1. Add Laplace noise (scale = 1/epsilon) to each probability value
 *   2. Clamp noised values to [0, Infinity) and re-normalize to sum to 1
 *   3. Convert exact age to decade bucket
 *   4. Strip timestamp to ISO date only
 *
 * @param classification - Raw classification output
 * @param demographics - Patient demographics (all optional)
 * @param measurement - Lesion measurement data (optional)
 * @param outcome - Clinical outcome if known (optional)
 * @param epsilon - Differential privacy parameter (default 1.0; lower = more noise)
 * @returns De-identified AnonymizedCase ready for brain sharing
 */
export function anonymizeCase(
	classification: {
		topClass: string;
		confidence: number;
		probabilities: Array<{ className: string; probability: number }>;
	},
	demographics: {
		age?: number;
		sex?: string;
		bodyLocation?: string;
	},
	measurement: { diameterMm?: number; abcdeScore?: number } | null,
	outcome: {
		concordant?: boolean;
		biopsied?: boolean;
		pathologyResult?: string;
	} | null,
	epsilon: number = DEFAULT_EPSILON
): AnonymizedCase {
	// 1. Add Laplace DP noise to probabilities and re-normalize
	const scale = 1.0 / epsilon;
	const noisedEntries: Array<[string, number]> = classification.probabilities.map(
		(p) => [p.className, Math.max(0, p.probability + laplaceSample(scale))]
	);

	const sum = noisedEntries.reduce((acc, [, v]) => acc + v, 0);
	const normalizedProbs: Record<string, number> = {};
	for (const [cls, val] of noisedEntries) {
		normalizedProbs[cls] = sum > 0 ? val / sum : 0;
	}

	// Noised confidence is the re-normalized value for the top class
	const noisedConfidence = normalizedProbs[classification.topClass] ?? 0;

	// 2. Reduce demographics
	const ageDecade = ageToDecade(demographics.age);
	const sex = demographics.sex ?? null;
	const bodyLocation = demographics.bodyLocation ?? "unknown";

	// 3. Derive outcome category
	let outcomeCategory: AnonymizedCase["outcome"] = null;
	if (outcome) {
		if (outcome.biopsied) {
			outcomeCategory = "biopsied";
		} else if (outcome.concordant === true) {
			outcomeCategory = "concordant";
		} else if (outcome.concordant === false) {
			// Concordant explicitly false means discordant -- determine direction
			// If the system missed something dangerous, it is "discordant_missed";
			// if it overcalled benign, it is "discordant_overcall".
			// Without further info, default to overcall (safer assumption).
			outcomeCategory = "discordant_overcall";
		}
	}

	// 4. Strip timestamp to date only (k-anonymity)
	const timestamp = stripToDateOnly(new Date().toISOString());

	return {
		probabilities: normalizedProbs,
		topClass: classification.topClass,
		confidence: noisedConfidence,
		bodyLocation,
		ageDecade,
		sex,
		diameterMm: measurement?.diameterMm ?? null,
		abcdeScore: measurement?.abcdeScore ?? null,
		outcome: outcomeCategory,
		pathologyResult: outcome?.pathologyResult ?? null,
		timestamp,
		modelVersion: "dragnes-v1",
	};
}
