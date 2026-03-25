/**
 * Consumer translation tests.
 * Verifies that all 7 HAM10000 classes translate to user-friendly results
 * with correct risk levels and action recommendations.
 */
import { describe, it, expect } from "vitest";
import { translateForConsumer } from "../src/lib/mela/consumer-translation";
import type { ConsumerRiskLevel } from "../src/lib/mela/consumer-translation";

const ALL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

describe("translateForConsumer", () => {
	it("maps all 7 classes to consumer-friendly text", () => {
		for (const cls of ALL_CLASSES) {
			const result = translateForConsumer(cls, 0.8);
			expect(result.headline).toBeTruthy();
			expect(result.explanation).toBeTruthy();
			expect(result.action).toBeTruthy();
			expect(result.medicalTerm).toBe(cls);
		}
	});

	it("assigns red risk to melanoma", () => {
		const result = translateForConsumer("mel", 0.9);
		expect(result.riskLevel).toBe("red");
		expect(result.urgency).toBe("urgent");
	});

	it("assigns green risk to nv (common mole)", () => {
		const result = translateForConsumer("nv", 0.85);
		expect(result.riskLevel).toBe("green");
	});

	it("assigns orange risk to bcc", () => {
		const result = translateForConsumer("bcc", 0.7);
		expect(result.riskLevel).toBe("orange");
	});

	it("assigns yellow risk to akiec", () => {
		const result = translateForConsumer("akiec", 0.6);
		expect(result.riskLevel).toBe("yellow");
	});

	it("sets shouldSeeDoctor for cancer classes", () => {
		for (const cls of ["mel", "bcc", "akiec"]) {
			expect(translateForConsumer(cls, 0.8).shouldSeeDoctor).toBe(true);
		}
	});

	it("sets shouldSeeDoctor=false for benign classes", () => {
		for (const cls of ["nv", "bkl", "df", "vasc"]) {
			expect(translateForConsumer(cls, 0.8).shouldSeeDoctor).toBe(false);
		}
	});

	it("adds low-confidence caveat when confidence < 0.4", () => {
		const result = translateForConsumer("nv", 0.2);
		expect(result.action).toContain("confidence is low");
	});

	it("upgrades green to yellow when cancer probs exceed 30%", () => {
		const probs = [
			{ className: "nv", probability: 0.5 },
			{ className: "mel", probability: 0.2 },
			{ className: "bcc", probability: 0.15 },
			{ className: "akiec", probability: 0.05 },
			{ className: "bkl", probability: 0.05 },
			{ className: "df", probability: 0.025 },
			{ className: "vasc", probability: 0.025 },
		];
		const result = translateForConsumer("nv", 0.5, probs);
		expect(result.riskLevel).toBe("yellow");
	});

	it("returns valid riskColor hex codes", () => {
		for (const cls of ALL_CLASSES) {
			const result = translateForConsumer(cls, 0.8);
			expect(result.riskColor).toMatch(/^#[0-9a-f]{6}$/);
		}
	});
});
