/**
 * ICD-10-CM code mapping for HAM10000 lesion classes.
 * Maps each classification to the appropriate billing/diagnostic code.
 */

export interface ICD10Code {
	code: string;
	description: string;
	category: "malignant" | "in_situ" | "benign" | "uncertain";
}

export const ICD10_MAP: Record<string, ICD10Code[]> = {
	mel: [
		{ code: "C43.9", description: "Malignant melanoma of skin, unspecified", category: "malignant" },
		{ code: "C43.0", description: "Malignant melanoma of lip", category: "malignant" },
		{ code: "C43.3", description: "Malignant melanoma of other/unspecified parts of face", category: "malignant" },
		{ code: "C43.5", description: "Malignant melanoma of trunk", category: "malignant" },
		{ code: "C43.6", description: "Malignant melanoma of upper limb", category: "malignant" },
		{ code: "C43.7", description: "Malignant melanoma of lower limb", category: "malignant" },
		{ code: "D03.9", description: "Melanoma in situ, unspecified", category: "in_situ" },
	],
	bcc: [
		{ code: "C44.91", description: "Basal cell carcinoma of skin, unspecified", category: "malignant" },
		{ code: "C44.01", description: "Basal cell carcinoma of skin of lip", category: "malignant" },
		{ code: "C44.31", description: "Basal cell carcinoma of skin of face", category: "malignant" },
		{ code: "C44.51", description: "Basal cell carcinoma of skin of trunk", category: "malignant" },
	],
	akiec: [
		{ code: "L57.0", description: "Actinic keratosis", category: "benign" },
		{ code: "D04.9", description: "Carcinoma in situ of skin, unspecified", category: "in_situ" },
	],
	bkl: [
		{ code: "L82.1", description: "Other seborrheic keratosis", category: "benign" },
		{ code: "L82.0", description: "Inflamed seborrheic keratosis", category: "benign" },
	],
	nv: [
		{ code: "D22.9", description: "Melanocytic nevi, unspecified", category: "benign" },
		{ code: "D22.5", description: "Melanocytic nevi of trunk", category: "benign" },
		{ code: "I78.1", description: "Nevus, non-neoplastic", category: "benign" },
	],
	df: [
		{ code: "D23.9", description: "Other benign neoplasm of skin, unspecified", category: "benign" },
		{ code: "L98.8", description: "Other specified disorders of skin (dermatofibroma)", category: "benign" },
	],
	vasc: [
		{ code: "D18.01", description: "Hemangioma of skin and subcutaneous tissue", category: "benign" },
		{ code: "I78.8", description: "Other diseases of capillaries", category: "benign" },
	],
};

/** Get primary ICD-10 code for a lesion class */
export function getPrimaryICD10(lesionClass: string): ICD10Code | null {
	return ICD10_MAP[lesionClass]?.[0] ?? null;
}

/** Get ICD-10 code adjusted for body location */
export function getLocationSpecificICD10(lesionClass: string, bodyLocation: string): ICD10Code {
	const codes = ICD10_MAP[lesionClass] || [];
	// Try to find location-specific code
	const locationTerms: Record<string, string[]> = {
		head: ["face", "lip", "scalp"],
		neck: ["face"],
		trunk: ["trunk"],
		upper_extremity: ["upper limb", "arm"],
		lower_extremity: ["lower limb", "leg"],
	};
	const terms = locationTerms[bodyLocation] || [];
	for (const code of codes) {
		for (const term of terms) {
			if (code.description.toLowerCase().includes(term)) return code;
		}
	}
	return codes[0] || { code: "R21", description: "Rash and other nonspecific skin eruption", category: "uncertain" as const };
}
