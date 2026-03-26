/**
 * Bayesian Risk Stratification Tests (ADR-129)
 *
 * Tests for the Bayesian post-test probability calculation,
 * age-adjusted prevalence, clinical history multipliers,
 * and risk tier assignment.
 */
import { describe, it, expect } from "vitest";
import { assessRisk } from "../src/lib/mela/risk-stratification";
import type { ClinicalHistory } from "../src/lib/mela/risk-stratification";

/** Build a full probability distribution with the given overrides */
function makeAllProbs(overrides: Partial<Record<string, number>>): Record<string, number> {
	const base: Record<string, number> = {
		akiec: 0.02,
		bcc: 0.02,
		bkl: 0.04,
		df: 0.02,
		mel: 0.02,
		nv: 0.84,
		vasc: 0.04,
	};
	for (const [k, v] of Object.entries(overrides)) {
		if (v !== undefined) base[k] = v;
	}
	return base;
}

describe("assessRisk", () => {
	describe("basic Bayesian update", () => {
		it("returns minimal risk for strongly benign prediction", () => {
			const result = assessRisk(
				"nv",
				0.90,
				makeAllProbs({ nv: 0.90, mel: 0.01, bcc: 0.01, akiec: 0.01 }),
			);
			expect(result.riskLevel).toBe("minimal");
			expect(result.postTestProbability).toBeLessThan(0.01);
		});

		it("returns high or very-high risk for strong melanoma prediction", () => {
			const result = assessRisk(
				"mel",
				0.85,
				makeAllProbs({ mel: 0.85, nv: 0.05, bcc: 0.05, akiec: 0.03 }),
			);
			// 93% combined malignant (mel+bcc+akiec), with base prevalence 0.10
			expect(["high", "very-high"]).toContain(result.riskLevel);
			expect(result.postTestProbability).toBeGreaterThan(0.20);
		});

		it("returns moderate risk for intermediate malignant probability", () => {
			// ~30% combined malignant probability
			const result = assessRisk(
				"mel",
				0.20,
				makeAllProbs({ mel: 0.20, bcc: 0.05, akiec: 0.05, nv: 0.60 }),
			);
			// 30% malignant combined, 10% prevalence -> LR ~0.43 -> post-test ~4.5%
			expect(["low", "moderate"]).toContain(result.riskLevel);
			expect(result.postTestProbability).toBeGreaterThan(0.01);
			expect(result.postTestProbability).toBeLessThan(0.20);
		});

		it("produces post-test probability between 0 and 1", () => {
			const result = assessRisk(
				"mel",
				0.50,
				makeAllProbs({ mel: 0.50, nv: 0.30, bcc: 0.10 }),
			);
			expect(result.postTestProbability).toBeGreaterThanOrEqual(0);
			expect(result.postTestProbability).toBeLessThanOrEqual(1);
		});

		it("includes all required fields in the result", () => {
			const result = assessRisk("nv", 0.80, makeAllProbs({ nv: 0.80 }));
			expect(result).toHaveProperty("riskLevel");
			expect(result).toHaveProperty("postTestProbability");
			expect(result).toHaveProperty("preTestProbability");
			expect(result).toHaveProperty("likelihoodRatio");
			expect(result).toHaveProperty("headline");
			expect(result).toHaveProperty("action");
			expect(result).toHaveProperty("urgency");
			expect(result).toHaveProperty("color");
		});
	});

	describe("age-based prevalence adjustment", () => {
		it("produces lower risk for young patients (< 30)", () => {
			const probs = makeAllProbs({ mel: 0.30, bcc: 0.05, akiec: 0.05 });
			const young = assessRisk("mel", 0.30, probs, { age: 25 });
			const baseline = assessRisk("mel", 0.30, probs);
			// Age multiplier for <30 is 0.3, which should lower pre-test probability
			expect(young.preTestProbability).toBeLessThan(baseline.preTestProbability);
			expect(young.postTestProbability).toBeLessThan(baseline.postTestProbability);
		});

		it("produces higher risk for elderly patients (> 70)", () => {
			const probs = makeAllProbs({ mel: 0.30, bcc: 0.05, akiec: 0.05 });
			const elderly = assessRisk("mel", 0.30, probs, { age: 75 });
			const baseline = assessRisk("mel", 0.30, probs);
			// Age multiplier for >70 is 2.5
			expect(elderly.preTestProbability).toBeGreaterThan(baseline.preTestProbability);
			expect(elderly.postTestProbability).toBeGreaterThan(baseline.postTestProbability);
		});

		it("returns multiplier of 1.0 for undefined age", () => {
			const probs = makeAllProbs({ mel: 0.30 });
			const noAge = assessRisk("mel", 0.30, probs);
			const explicitNoAge = assessRisk("mel", 0.30, probs, {});
			expect(noAge.preTestProbability).toBe(explicitNoAge.preTestProbability);
		});

		it("returns multiplier of 1.0 for negative age", () => {
			const probs = makeAllProbs({ mel: 0.30 });
			const negAge = assessRisk("mel", 0.30, probs, { age: -5 });
			const noAge = assessRisk("mel", 0.30, probs);
			expect(negAge.preTestProbability).toBe(noAge.preTestProbability);
		});
	});

	describe("clinical history adjustments", () => {
		const probs = makeAllProbs({ mel: 0.30, bcc: 0.05, akiec: 0.05 });

		it("increases risk for new, changing lesion with family history", () => {
			const history: ClinicalHistory = {
				isNew: "new",
				hasChanged: "yes",
				familyHistoryMelanoma: "yes",
			};
			const withHistory = assessRisk("mel", 0.30, probs, undefined, history);
			const baseline = assessRisk("mel", 0.30, probs);
			expect(withHistory.preTestProbability).toBeGreaterThan(baseline.preTestProbability);
			expect(withHistory.postTestProbability).toBeGreaterThan(baseline.postTestProbability);
		});

		it("slightly decreases prevalence for previously biopsied lesions", () => {
			const history: ClinicalHistory = {
				previouslyBiopsied: "yes",
			};
			const biopsied = assessRisk("mel", 0.30, probs, undefined, history);
			const baseline = assessRisk("mel", 0.30, probs);
			expect(biopsied.preTestProbability).toBeLessThan(baseline.preTestProbability);
		});

		it("increases risk for bleeding symptom", () => {
			const history: ClinicalHistory = {
				symptoms: ["bleeding"],
			};
			const withBleeding = assessRisk("mel", 0.30, probs, undefined, history);
			const baseline = assessRisk("mel", 0.30, probs);
			expect(withBleeding.preTestProbability).toBeGreaterThan(baseline.preTestProbability);
		});

		it("does not change risk when no clinical history provided", () => {
			const withUndefined = assessRisk("mel", 0.30, probs, undefined, undefined);
			const baseline = assessRisk("mel", 0.30, probs);
			expect(withUndefined.preTestProbability).toBe(baseline.preTestProbability);
		});

		it("does not change risk when all fields are unsure/none", () => {
			const history: ClinicalHistory = {
				isNew: "unsure",
				hasChanged: "unsure",
				familyHistoryMelanoma: "unsure",
				symptoms: ["none"],
			};
			const unsure = assessRisk("mel", 0.30, probs, undefined, history);
			const baseline = assessRisk("mel", 0.30, probs);
			// "unsure" and "none" should produce multiplier of 1.0
			expect(unsure.preTestProbability).toBe(baseline.preTestProbability);
		});
	});

	describe("edge cases", () => {
		it("handles zero malignant probability without errors", () => {
			const probs = makeAllProbs({ mel: 0, bcc: 0, akiec: 0, nv: 0.90 });
			const result = assessRisk("nv", 0.90, probs);
			expect(result.riskLevel).toBe("minimal");
			expect(result.postTestProbability).toBeGreaterThanOrEqual(0);
		});

		it("handles near-100% malignant probability without exceeding 1", () => {
			const probs = makeAllProbs({ mel: 0.95, bcc: 0.03, akiec: 0.01 });
			const result = assessRisk("mel", 0.95, probs);
			expect(result.postTestProbability).toBeLessThanOrEqual(1);
			expect(result.riskLevel).toBe("very-high");
		});

		it("caps adjusted prevalence at 0.5 even with extreme multipliers", () => {
			const probs = makeAllProbs({ mel: 0.50 });
			const history: ClinicalHistory = {
				isNew: "new",
				hasChanged: "yes",
				familyHistoryMelanoma: "yes",
				symptoms: ["bleeding", "itching", "pain"],
			};
			// Age 80 (2.5x) + aggressive clinical history => large multiplier
			const result = assessRisk("mel", 0.50, probs, { age: 80 }, history);
			expect(result.preTestProbability).toBeLessThanOrEqual(0.5);
		});

		it("assigns correct risk tier colors", () => {
			// Very high risk
			const veryHigh = assessRisk("mel", 0.95, makeAllProbs({ mel: 0.95, bcc: 0.03, akiec: 0.01 }));
			expect(veryHigh.color).toBe("#ef4444");

			// Minimal risk
			const minimal = assessRisk("nv", 0.95, makeAllProbs({ nv: 0.95, mel: 0.01 }));
			expect(minimal.color).toBe("#14b8a6");
		});

		it("likelihood ratio is positive for all inputs", () => {
			const result = assessRisk("mel", 0.50, makeAllProbs({ mel: 0.50 }));
			expect(result.likelihoodRatio).toBeGreaterThan(0);
		});
	});
});
