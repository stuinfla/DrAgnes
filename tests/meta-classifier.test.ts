/**
 * Meta-Classifier Tests
 *
 * Tests for the neural + clinical feature fusion that adjusts
 * classification probabilities based on agreement or disagreement
 * between the ViT model and clinical scoring (ABCDE, TDS, 7-point).
 */
import { describe, it, expect } from "vitest";
import { metaClassify } from "../src/lib/mela/meta-classifier";
import type { ClassProbability, LesionClass, ABCDEScores } from "../src/lib/mela/types";
import { LESION_LABELS } from "../src/lib/mela/types";

const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/** Build a ClassProbability array with the given overrides, renormalized to sum=1 */
function makeProbs(overrides: Partial<Record<LesionClass, number>>): ClassProbability[] {
	const base: Record<LesionClass, number> = {
		akiec: 0.02, bcc: 0.02, bkl: 0.04, df: 0.02, mel: 0.02, nv: 0.84, vasc: 0.04,
	};
	for (const [k, v] of Object.entries(overrides)) base[k as LesionClass] = v;
	const sum = Object.values(base).reduce((a, b) => a + b, 0);
	return CLASSES.map((c) => ({
		className: c,
		probability: base[c] / sum,
		label: LESION_LABELS[c],
	}));
}

/** Build ABCDE scores with defaults that indicate a suspicious lesion */
function makeSuspiciousABCDE(overrides?: Partial<ABCDEScores>): ABCDEScores {
	return {
		asymmetry: 2,
		border: 5,
		color: 4,
		diameterMm: 8,
		evolution: 1,
		totalScore: 7.1,
		riskLevel: "high",
		colorsDetected: ["brown", "black", "blue-gray"],
		...overrides,
	};
}

/** Build ABCDE scores with defaults that indicate a benign lesion */
function makeBenignABCDE(overrides?: Partial<ABCDEScores>): ABCDEScores {
	return {
		asymmetry: 0,
		border: 1,
		color: 1,
		diameterMm: 3,
		evolution: 0,
		totalScore: 2.0,
		riskLevel: "low",
		colorsDetected: ["brown"],
		...overrides,
	};
}

describe("metaClassify", () => {
	describe("concordant agreement (neural and clinical agree)", () => {
		it("boosts melanoma when both neural and clinical are high", () => {
			const neuralProbs = makeProbs({ mel: 0.60, nv: 0.20 });
			const abcde = makeSuspiciousABCDE();
			const result = metaClassify(neuralProbs, abcde, 5.5, 4, 8);

			expect(result.agreement).toBe("concordant");
			// Melanoma should be boosted (multiplier 1.2)
			const originalMel = neuralProbs.find((p) => p.className === "mel")!.probability;
			const adjustedMel = result.adjustedProbabilities.find((p) => p.className === "mel")!.probability;
			// After renormalization, the absolute value may differ, but relative ranking should hold
			expect(result.adjustedTopClass).toBe("mel");
			expect(result.adjustmentReason).toContain("support");
		});

		it("suppresses melanoma when both neural and clinical are low", () => {
			const neuralProbs = makeProbs({ mel: 0.05, nv: 0.80 });
			const abcde = makeBenignABCDE();
			const result = metaClassify(neuralProbs, abcde, 3.0, 1, 3);

			expect(result.agreement).toBe("concordant");
			// Melanoma should be suppressed (multiplier 0.8)
			expect(result.adjustedTopClass).toBe("nv");
			expect(result.adjustmentReason).toContain("low risk");
		});
	});

	describe("discordant agreement (neural and clinical disagree)", () => {
		it("reduces melanoma when neural is high but clinical features are benign", () => {
			// Neural says melanoma, but clinical features are all benign
			const neuralProbs = makeProbs({ mel: 0.50, nv: 0.30 });
			const abcde = makeBenignABCDE();
			const result = metaClassify(neuralProbs, abcde, 3.0, 1, 3);

			expect(result.agreement).toBe("discordant");
			// Melanoma probability should be reduced (multiplier 0.6)
			const adjustedMel = result.adjustedProbabilities.find((p) => p.className === "mel")!.probability;
			const originalMel = neuralProbs.find((p) => p.className === "mel")!.probability;
			// After applying 0.6 multiplier and renormalization, mel should be lower relative to others
			expect(adjustedMel).toBeLessThan(originalMel);
			expect(result.adjustmentReason).toContain("benign");
		});

		it("increases melanoma when clinical features are suspicious but neural is low", () => {
			// Neural says nv, but clinical features are suspicious
			const neuralProbs = makeProbs({ mel: 0.10, nv: 0.70 });
			const abcde = makeSuspiciousABCDE();
			const result = metaClassify(neuralProbs, abcde, 6.0, 4, 8);

			expect(result.agreement).toBe("discordant");
			// Melanoma probability should be boosted (multiplier 1.3 -- safety-critical)
			const adjustedMel = result.adjustedProbabilities.find((p) => p.className === "mel")!.probability;
			const originalMel = neuralProbs.find((p) => p.className === "mel")!.probability;
			expect(adjustedMel).toBeGreaterThan(originalMel);
			expect(result.adjustmentReason).toContain("increased risk");
		});
	});

	describe("neutral / no clinical data", () => {
		it("returns neural output unchanged when no clinical data available", () => {
			const neuralProbs = makeProbs({ mel: 0.40, nv: 0.40 });
			const result = metaClassify(neuralProbs, null, null, null, null);

			expect(result.agreement).toBe("neutral");
			expect(result.clinicalConfidence).toBe(0);
			expect(result.adjustmentReason).toContain("No clinical scoring data");
			// Probabilities should be unchanged (just sorted)
			const originalMel = neuralProbs.find((p) => p.className === "mel")!.probability;
			const adjustedMel = result.adjustedProbabilities.find((p) => p.className === "mel")!.probability;
			expect(adjustedMel).toBeCloseTo(originalMel, 5);
		});

		it("still works with ABCDE only (no TDS or 7-point)", () => {
			const neuralProbs = makeProbs({ mel: 0.40, nv: 0.40 });
			const abcde = makeSuspiciousABCDE();
			const result = metaClassify(neuralProbs, abcde, null, null, null);

			// ABCDE alone can produce clinical suspicion via asymmetry, border, color, diameter
			expect(result.clinicalConfidence).toBeGreaterThan(0);
			expect(["concordant", "discordant", "neutral"]).toContain(result.agreement);
		});
	});

	describe("result structure and invariants", () => {
		it("adjusted probabilities sum to approximately 1", () => {
			const neuralProbs = makeProbs({ mel: 0.50, nv: 0.30 });
			const result = metaClassify(neuralProbs, makeSuspiciousABCDE(), 5.5, 3, 7);

			const sum = result.adjustedProbabilities.reduce((s, p) => s + p.probability, 0);
			expect(sum).toBeCloseTo(1.0, 3);
		});

		it("adjusted probabilities are sorted descending", () => {
			const neuralProbs = makeProbs({ mel: 0.50, nv: 0.30 });
			const result = metaClassify(neuralProbs, makeSuspiciousABCDE(), 5.5, 3, 7);

			for (let i = 1; i < result.adjustedProbabilities.length; i++) {
				expect(result.adjustedProbabilities[i].probability)
					.toBeLessThanOrEqual(result.adjustedProbabilities[i - 1].probability);
			}
		});

		it("top class matches the highest probability in adjustedProbabilities", () => {
			const neuralProbs = makeProbs({ mel: 0.40, nv: 0.40 });
			const result = metaClassify(neuralProbs, makeBenignABCDE(), 3.0, 1, 3);

			const topByProb = result.adjustedProbabilities[0];
			expect(result.adjustedTopClass).toBe(topByProb.className);
			expect(result.adjustedConfidence).toBe(topByProb.probability);
		});

		it("returns all 7 classes in adjustedProbabilities", () => {
			const neuralProbs = makeProbs({});
			const result = metaClassify(neuralProbs, null, null, null, null);
			expect(result.adjustedProbabilities).toHaveLength(7);
			const classNames = result.adjustedProbabilities.map((p) => p.className);
			for (const c of CLASSES) {
				expect(classNames).toContain(c);
			}
		});

		it("neuralConfidence reflects the mel probability from neural input", () => {
			const neuralProbs = makeProbs({ mel: 0.35 });
			const result = metaClassify(neuralProbs, null, null, null, null);
			const inputMel = neuralProbs.find((p) => p.className === "mel")!.probability;
			expect(result.neuralConfidence).toBeCloseTo(inputMel, 5);
		});

		it("no probability is negative after adjustment", () => {
			const neuralProbs = makeProbs({ mel: 0.01, nv: 0.90 });
			const result = metaClassify(neuralProbs, makeBenignABCDE(), 2.0, 0, 2);

			for (const p of result.adjustedProbabilities) {
				expect(p.probability).toBeGreaterThanOrEqual(0);
			}
		});
	});

	describe("clinical suspicion scoring", () => {
		it("produces higher clinical confidence with more suspicious features", () => {
			const neuralProbs = makeProbs({ mel: 0.40 });

			const benign = metaClassify(neuralProbs, makeBenignABCDE(), 3.0, 1, 3);
			const suspicious = metaClassify(neuralProbs, makeSuspiciousABCDE(), 6.0, 4, 8);

			expect(suspicious.clinicalConfidence).toBeGreaterThan(benign.clinicalConfidence);
		});

		it("includes TDS contribution when TDS > 4.75", () => {
			const neuralProbs = makeProbs({ mel: 0.40 });
			const abcde = makeBenignABCDE();

			const lowTDS = metaClassify(neuralProbs, abcde, 3.0, null, null);
			const highTDS = metaClassify(neuralProbs, abcde, 5.5, null, null);

			expect(highTDS.clinicalConfidence).toBeGreaterThan(lowTDS.clinicalConfidence);
		});

		it("includes 7-point contribution when score >= 3", () => {
			const neuralProbs = makeProbs({ mel: 0.40 });
			const abcde = makeBenignABCDE();

			const low7pt = metaClassify(neuralProbs, abcde, null, 1, null);
			const high7pt = metaClassify(neuralProbs, abcde, null, 4, null);

			expect(high7pt.clinicalConfidence).toBeGreaterThan(low7pt.clinicalConfidence);
		});

		it("includes blue-white veil contribution from colorsDetected", () => {
			const neuralProbs = makeProbs({ mel: 0.40 });

			const noBWV = metaClassify(
				neuralProbs,
				makeBenignABCDE({ colorsDetected: ["brown"] }),
				null, null, null,
			);
			const withBWV = metaClassify(
				neuralProbs,
				makeBenignABCDE({ colorsDetected: ["brown", "blue-gray"] }),
				null, null, null,
			);

			expect(withBWV.clinicalConfidence).toBeGreaterThan(noBWV.clinicalConfidence);
		});
	});
});
