/**
 * ICD-10 code mapping tests.
 * Verifies all 7 HAM10000 classes have codes and location-specific lookup works.
 */
import { describe, it, expect } from "vitest";
import { getPrimaryICD10, getLocationSpecificICD10, ICD10_MAP } from "../src/lib/dragnes/icd10";
import type { ICD10Code } from "../src/lib/dragnes/icd10";

const ALL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

describe("ICD10_MAP", () => {
	it("has entries for all 7 classes", () => {
		for (const cls of ALL_CLASSES) {
			expect(ICD10_MAP[cls]).toBeDefined();
			expect(ICD10_MAP[cls].length).toBeGreaterThan(0);
		}
	});

	it("every code has proper format (letter + digits)", () => {
		for (const cls of ALL_CLASSES) {
			for (const entry of ICD10_MAP[cls]) {
				expect(entry.code).toMatch(/^[A-Z]\d{2}/);
				expect(entry.description).toBeTruthy();
				expect(["malignant", "in_situ", "benign", "uncertain"]).toContain(entry.category);
			}
		}
	});
});

describe("getPrimaryICD10", () => {
	it("returns the first code for each known class", () => {
		for (const cls of ALL_CLASSES) {
			const code = getPrimaryICD10(cls);
			expect(code).not.toBeNull();
			expect(code!.code).toBe(ICD10_MAP[cls][0].code);
		}
	});

	it("returns null for unknown class", () => {
		expect(getPrimaryICD10("unknown_class")).toBeNull();
	});

	it("mel primary is C43.9", () => {
		expect(getPrimaryICD10("mel")!.code).toBe("C43.9");
	});

	it("bcc primary is C44.91", () => {
		expect(getPrimaryICD10("bcc")!.code).toBe("C44.91");
	});
});

describe("getLocationSpecificICD10", () => {
	it("returns trunk-specific code for mel on trunk", () => {
		const code = getLocationSpecificICD10("mel", "trunk");
		expect(code.code).toBe("C43.5");
		expect(code.description).toContain("trunk");
	});

	it("returns lip-specific code for bcc on head", () => {
		// "head" maps to terms ["face","lip","scalp"]; lip is matched first (C44.01)
		const code = getLocationSpecificICD10("bcc", "head");
		expect(code.code).toBe("C44.01");
		expect(code.description.toLowerCase()).toContain("lip");
	});

	it("returns upper-limb code for mel on upper_extremity", () => {
		const code = getLocationSpecificICD10("mel", "upper_extremity");
		expect(code.code).toBe("C43.6");
	});

	it("falls back to primary code for unmapped location", () => {
		const code = getLocationSpecificICD10("mel", "palms_soles");
		expect(code.code).toBe(ICD10_MAP.mel[0].code);
	});

	it("returns fallback R21 for totally unknown class", () => {
		const code = getLocationSpecificICD10("nonexistent", "trunk");
		expect(code.code).toBe("R21");
	});
});
