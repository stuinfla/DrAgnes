/**
 * Threshold-based classification tests (ADR-123).
 * Verifies per-class threshold filtering, mode selection, and melanoma safety.
 */
import { describe, it, expect } from "vitest";
import { applyThresholds, getThresholds } from "../src/lib/mela/threshold-classifier";
import type { ClassProbability, LesionClass } from "../src/lib/mela/types";
import { LESION_LABELS } from "../src/lib/mela/types";

const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

function makeProbs(overrides: Partial<Record<LesionClass, number>>): ClassProbability[] {
	const base: Record<LesionClass, number> = { akiec: 0.02, bcc: 0.02, bkl: 0.02, df: 0.02, mel: 0.02, nv: 0.86, vasc: 0.04 };
	for (const [k, v] of Object.entries(overrides)) base[k as LesionClass] = v;
	// renormalize
	const sum = Object.values(base).reduce((a, b) => a + b, 0);
	return CLASSES.map((c) => ({ className: c, probability: base[c] / sum, label: LESION_LABELS[c] }));
}

describe("applyThresholds", () => {
	it("returns mel as top class when mel exceeds its threshold", () => {
		const probs = makeProbs({ mel: 0.70, nv: 0.15 });
		const result = applyThresholds(probs, "default");
		expect(result.topClass).toBe("mel");
		expect(result.confidence).toBeGreaterThan(0.5);
	});

	it("falls back to argmax when nothing exceeds threshold", () => {
		// All probabilities far below their thresholds
		const probs: ClassProbability[] = CLASSES.map((c) => ({
			className: c, probability: 0.001, label: LESION_LABELS[c],
		}));
		probs.find((p) => p.className === "nv")!.probability = 0.01;
		const result = applyThresholds(probs, "default");
		expect(result.topClass).toBe("nv"); // highest raw value
	});

	it("uses screening thresholds when mode is screening", () => {
		const thresholds = getThresholds("screening");
		expect(thresholds.mel).toBe(0.25);
		const probs = makeProbs({ mel: 0.30, nv: 0.50 });
		const result = applyThresholds(probs, "screening");
		// mel exceeds 0.25 in screening, nv also exceeds 0.15; highest wins
		expect(["mel", "nv"]).toContain(result.topClass);
	});

	it("uses triage thresholds (same as default)", () => {
		const def = getThresholds("default");
		const tri = getThresholds("triage");
		for (const c of CLASSES) expect(def[c]).toBe(tri[c]);
	});

	it("preserves melanoma safety: mel above threshold always surfaces", () => {
		// mel threshold = 0.6204; set mel=0.65 directly with nv low
		const probs: ClassProbability[] = [
			{ className: "mel", probability: 0.65, label: "Melanoma" },
			{ className: "nv", probability: 0.15, label: "Melanocytic Nevus" },
			{ className: "bkl", probability: 0.10, label: "Benign Keratosis" },
			{ className: "bcc", probability: 0.04, label: "Basal Cell Carcinoma" },
			{ className: "akiec", probability: 0.03, label: "Actinic Keratosis" },
			{ className: "df", probability: 0.02, label: "Dermatofibroma" },
			{ className: "vasc", probability: 0.01, label: "Vascular Lesion" },
		];
		const result = applyThresholds(probs, "default");
		expect(result.topClass).toBe("mel");
	});

	it("includes thresholdMode in output", () => {
		const probs = makeProbs({});
		expect(applyThresholds(probs, "screening").thresholdMode).toBe("screening");
		expect(applyThresholds(probs, "triage").thresholdMode).toBe("triage");
		expect(applyThresholds(probs, "default").thresholdMode).toBe("default");
	});

	it("returns all 7 probabilities sorted descending", () => {
		const probs = makeProbs({ mel: 0.5 });
		const result = applyThresholds(probs, "default");
		expect(result.probabilities).toHaveLength(7);
		for (let i = 1; i < result.probabilities.length; i++) {
			expect(result.probabilities[i - 1].probability).toBeGreaterThanOrEqual(result.probabilities[i].probability);
		}
	});
});
