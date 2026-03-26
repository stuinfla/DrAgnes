/**
 * Clinical baselines tests.
 * Verifies TDS formula, risk level thresholds, and 7-point checklist scoring.
 */
import { describe, it, expect } from "vitest";
import {
	computeTDS,
	tdsRiskLevel,
	computeSevenPointScore,
	TDS,
} from "../src/lib/mela/clinical-baselines";

describe("computeTDS", () => {
	it("computes A*1.3 + B*0.1 + C*0.5 + D*0.5", () => {
		// A=2, B=4, C=3, D=2 => 2*1.3 + 4*0.1 + 3*0.5 + 2*0.5 = 2.6+0.4+1.5+1.0 = 5.5
		expect(computeTDS(2, 4, 3, 2)).toBeCloseTo(5.5);
	});

	it("returns 0 for all-zero inputs", () => {
		expect(computeTDS(0, 0, 0, 0)).toBe(0);
	});

	it("uses the exported weight constants", () => {
		const result = computeTDS(1, 1, 1, 1);
		expect(result).toBeCloseTo(TDS.weights.A + TDS.weights.B + TDS.weights.C + TDS.weights.D);
	});
});

describe("tdsRiskLevel", () => {
	it("returns low for TDS < 4.75", () => {
		expect(tdsRiskLevel(3.0)).toBe("low");
		expect(tdsRiskLevel(4.74)).toBe("low");
	});

	it("returns moderate for TDS 4.75-5.45", () => {
		expect(tdsRiskLevel(4.75)).toBe("moderate");
		expect(tdsRiskLevel(5.0)).toBe("moderate");
		expect(tdsRiskLevel(5.45)).toBe("moderate");
	});

	it("returns high for TDS 5.46-7.0", () => {
		expect(tdsRiskLevel(5.46)).toBe("high");
		expect(tdsRiskLevel(7.0)).toBe("high");
	});

	it("returns critical for TDS > 7.0", () => {
		expect(tdsRiskLevel(7.01)).toBe("critical");
		expect(tdsRiskLevel(10.0)).toBe("critical");
	});
});

describe("computeSevenPointScore", () => {
	it("returns 0 with no structures present", () => {
		const r = computeSevenPointScore({
			hasIrregularNetwork: false, hasBlueWhiteVeil: false,
			hasStreaks: false, hasIrregularGlobules: false, hasRegressionStructures: false,
		});
		expect(r.score).toBe(0);
		expect(r.recommendation).toContain("No immediate concern");
	});

	it("scores major criteria at 2 points each", () => {
		const r = computeSevenPointScore({
			hasIrregularNetwork: true, hasBlueWhiteVeil: true,
			hasStreaks: false, hasIrregularGlobules: false, hasRegressionStructures: false,
		});
		expect(r.score).toBe(4);
		expect(r.recommendation).toContain("Professional evaluation may be warranted");
	});

	it("scores minor criteria at 1 point each", () => {
		const r = computeSevenPointScore({
			hasIrregularNetwork: false, hasBlueWhiteVeil: false,
			hasStreaks: true, hasIrregularGlobules: true, hasRegressionStructures: true,
		});
		expect(r.score).toBe(3);
	});

	it("recommends close monitoring for score of 2", () => {
		const r = computeSevenPointScore({
			hasIrregularNetwork: true, hasBlueWhiteVeil: false,
			hasStreaks: false, hasIrregularGlobules: false, hasRegressionStructures: false,
		});
		expect(r.score).toBe(2);
		expect(r.recommendation).toContain("Close monitoring");
	});

	it("includes detail strings for each detected structure", () => {
		const r = computeSevenPointScore({
			hasIrregularNetwork: true, hasBlueWhiteVeil: false,
			hasStreaks: true, hasIrregularGlobules: false, hasRegressionStructures: false,
		});
		expect(r.details).toHaveLength(2);
		expect(r.details[0]).toContain("Atypical pigment network");
		expect(r.details[1]).toContain("Irregular streaks");
	});
});
