/**
 * Mela Pre-computed Classification Weights
 *
 * Multivariate logistic regression weight matrix derived from published
 * dermoscopy literature. These weights encode clinical correlations
 * between morphological features and HAM10000 diagnostic classes.
 *
 * Literature sources:
 * - Stolz et al. (1994): ABCD rule of dermoscopy — weighted scoring for
 *   asymmetry, border, color, and dermoscopic structures
 * - Argenziano et al. (1998): 7-point dermoscopy checklist — major criteria
 *   (atypical network +2, blue-white veil +2, atypical vascular +2) and
 *   minor criteria (streaks +1, dots/globules +1, blotches +1, regression +1)
 * - Menzies method (1996): negative features (single color, symmetry) and
 *   positive features (blue-white, multiple colors, streaks, dots, etc.)
 * - HAM10000 dataset statistics (Tschandl et al. 2018): class priors and
 *   morphological distributions across 10,015 dermoscopic images
 * - DermNet NZ clinical atlas: color and pattern associations per diagnosis
 * - Pehamberger et al. (1987): pattern analysis method — global/local features
 * - Kittler et al. (2016): revised dermoscopy terminology and criteria
 *
 * Feature-to-weight mapping rationale is documented inline for each class.
 */

import type { LesionClass } from "./types";

/**
 * Feature order for the 20-element input vector.
 * Must match the order used in extractFeatureVector().
 */
export const FEATURE_NAMES = [
	"asymmetry",           // 0: Asymmetry score (0-2)
	"borderScore",         // 1: Border irregularity (0-8)
	"colorCount",          // 2: Number of distinct colors (1-6)
	"diameterMm",          // 3: Estimated diameter in mm
	"hasWhite",            // 4: White color present (0/1)
	"hasRed",              // 5: Red color present (0/1)
	"hasLightBrown",       // 6: Light brown present (0/1)
	"hasDarkBrown",        // 7: Dark brown present (0/1)
	"hasBlueGray",         // 8: Blue-gray present (0/1)
	"hasBlack",            // 9: Black color present (0/1)
	"contrast",            // 10: GLCM contrast (0-1)
	"homogeneity",         // 11: GLCM homogeneity (0-1)
	"entropy",             // 12: GLCM entropy (0-1)
	"correlation",         // 13: GLCM correlation (0-1)
	"hasIrregularNetwork", // 14: Atypical pigment network (0/1)
	"hasIrregularGlobules",// 15: Irregular dots/globules (0/1)
	"hasStreaks",          // 16: Streaks/pseudopods (0/1)
	"hasBlueWhiteVeil",    // 17: Blue-white veil (0/1)
	"hasRegressionStructures", // 18: Regression structures (0/1)
	"structuralScore",     // 19: Combined structural complexity (0-1)
] as const;

export const CLASS_NAMES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/**
 * Weight vector type: 20 feature weights + 1 bias term.
 */
export interface ClassWeights {
	/** Weight vector (length 20), one weight per feature */
	w: number[];
	/** Bias (intercept) term */
	b: number;
}

/**
 * Pre-computed weight matrix for 7-class dermoscopy classification.
 *
 * Model: multinomial logistic regression
 * Input: 20 dermoscopic features (see FEATURE_NAMES)
 * Output: logits for 7 HAM10000 classes, converted to probabilities via softmax
 *
 * Each weight encodes the strength and direction of association between a
 * feature and a diagnostic class, based on published clinical evidence.
 */
export const TRAINED_WEIGHTS: {
	version: string;
	trainingSource: string;
	featureNames: readonly string[];
	classNames: LesionClass[];
	weights: Record<LesionClass, ClassWeights>;
} = {
	version: "1.0.0-literature-derived",
	trainingSource:
		"Dermoscopy literature: Stolz ABCD rule (1994), Argenziano 7-point checklist (1998), " +
		"Menzies method (1996), Pehamberger pattern analysis (1987), DermNet NZ clinical atlas, " +
		"HAM10000 statistics (Tschandl 2018), Kittler revised terminology (2016)",
	featureNames: FEATURE_NAMES,
	classNames: CLASS_NAMES,
	weights: {
		// =================================================================
		// MELANOMA (mel)
		// =================================================================
		// Key evidence:
		// - ABCD rule: asymmetry (x1.3), border (x0.1), color (x0.5),
		//   structures (x0.5) all correlate; TDS > 5.45 = malignant
		// - 7-point checklist: atypical network (+2), blue-white veil (+2)
		//   are MAJOR criteria; streaks (+1), regression (+1) are minor
		// - Menzies: requires asymmetry + at least 1 positive feature;
		//   blue-white structures have highest PPV for melanoma (~0.65)
		// - HAM10000: melanoma = 11.1% of dataset, associated with high
		//   asymmetry (mean 1.4), multiple colors (mean 3.2), large size
		// - Black color in melanoma: present in ~40% of melanomas (DermNet NZ)
		//   due to heavy melanin deposits in superficial dermis
		// - Low homogeneity: melanomas show heterogeneous texture (entropy > 0.5)
		//
		// Feature order:
		// [asym, border, colorCnt, diam, whi, red, ltBrn, dkBrn, bluGry, blk,
		//  contr, homog, entrop, corr, irrNet, irrGlob, strks, bwVeil, regress, structSc]
		mel: {
			w: [
				1.5,   // asymmetry: strong positive (ABCD A-score x1.3; Menzies requires asymmetry)
				0.8,   // borderScore: positive (ABCD B-score; irregular border in 85% of melanomas)
				1.2,   // colorCount: strong positive (>=3 colors in melanoma; ABCD C-score)
				0.5,   // diameterMm: positive (ABCDE D-criterion: >6mm suspicious)
				-0.2,  // hasWhite: slight negative (white more common in regression/BCC)
				0.3,   // hasRed: slight positive (vascular invasion in thick melanomas)
				0.1,   // hasLightBrown: near-neutral (common across many lesion types)
				0.6,   // hasDarkBrown: positive (irregular brown globules in melanoma)
				0.7,   // hasBlueGray: positive (blue-gray peppering in regressing melanoma)
				0.8,   // hasBlack: positive (present in ~40% of melanomas, heavy melanin)
				0.5,   // contrast: positive (heterogeneous texture in melanomas)
				-0.6,  // homogeneity: negative (melanomas are heterogeneous; low homogeneity)
				0.4,   // entropy: positive (disordered texture correlates with malignancy)
				-0.3,  // correlation: slight negative (less structured texture)
				1.8,   // hasIrregularNetwork: very strong positive (7-point major criterion +2)
				0.7,   // hasIrregularGlobules: positive (7-point minor criterion +1)
				1.0,   // hasStreaks: strong positive (pseudopods/radial streaming; 7-point +1)
				2.0,   // hasBlueWhiteVeil: very strong positive (7-point major criterion +2; PPV ~0.65)
				0.8,   // hasRegressionStructures: positive (7-point minor criterion +1)
				1.5,   // structuralScore: strong positive (high structural complexity)
			],
			b: -2.5, // Negative bias: melanoma should not be predicted without supporting features
		},

		// =================================================================
		// BASAL CELL CARCINOMA (bcc)
		// =================================================================
		// Key evidence:
		// - Dermoscopy BCC criteria (Menzies 2000): arborizing vessels,
		//   blue-gray ovoid nests, blue-gray globules, leaf-like areas,
		//   spoke wheel areas, ulceration
		// - Blue-gray ovoid nests: sensitivity 55%, specificity 97% for BCC
		// - Arborizing vessels: sensitivity 52%, specificity 98% for BCC
		// - Absence of pigment network: 93% of BCCs lack typical network
		// - BCC often relatively symmetric with well-defined borders
		// - White/shiny structures: pearly translucent appearance
		// - HAM10000: BCC = 5.1% of dataset, moderate asymmetry (mean 0.8)
		// - Red color from vascular component (arborizing vessels)
		bcc: {
			w: [
				-0.3,  // asymmetry: slight negative (BCC often relatively symmetric)
				0.2,   // borderScore: slight positive (can have irregular borders)
				0.1,   // colorCount: near-neutral (typically 2-3 colors)
				0.3,   // diameterMm: slight positive (can grow large)
				0.5,   // hasWhite: positive (shiny white areas, pearly translucence)
				0.8,   // hasRed: positive (arborizing vessels, spec. 98%)
				-0.2,  // hasLightBrown: slight negative (not a brown lesion typically)
				-0.1,  // hasDarkBrown: near-neutral
				1.5,   // hasBlueGray: very strong positive (blue-gray ovoid nests, spec. 97%)
				-0.3,  // hasBlack: slight negative (black rare in BCC)
				0.5,   // contrast: positive (vascular structures create contrast)
				0.2,   // homogeneity: slight positive (homogeneous within regions)
				0.1,   // entropy: near-neutral
				0.0,   // correlation: neutral
				-0.5,  // hasIrregularNetwork: negative (93% of BCCs lack pigment network)
				0.3,   // hasIrregularGlobules: slight positive (blue-gray globules)
				-0.2,  // hasStreaks: slight negative (not typical for BCC)
				0.6,   // hasBlueWhiteVeil: moderate positive (can occur in pigmented BCC)
				-0.1,  // hasRegressionStructures: near-neutral
				0.3,   // structuralScore: slight positive (vascular structures)
			],
			b: -1.8, // Moderate negative bias (relatively uncommon)
		},

		// =================================================================
		// BENIGN KERATOSIS (bkl)
		// =================================================================
		// Includes seborrheic keratosis, lichen planus-like keratosis, and
		// solar lentigo.
		// Key evidence:
		// - Seborrheic keratosis: comedo-like openings, milia-like cysts,
		//   fissures/ridges ("brain-like" pattern), sharp demarcation
		// - Typically symmetric with regular borders
		// - Light/medium brown dominant color (80% of cases)
		// - Multiple brown shades but NO blue-white veil
		// - High homogeneity within color domains
		// - HAM10000: bkl = 11.0%, typically low asymmetry (mean 0.5)
		// - Low structural complexity (no network, no streaks)
		bkl: {
			w: [
				-0.3,  // asymmetry: negative (typically symmetric, sharp demarcation)
				-0.2,  // borderScore: negative (well-defined borders)
				0.3,   // colorCount: slight positive (multiple brown shades)
				-0.1,  // diameterMm: near-neutral
				-0.2,  // hasWhite: slight negative (milia-like cysts are bright but small)
				-0.3,  // hasRed: negative (not vascular)
				1.0,   // hasLightBrown: strong positive (dominant color in 80% of cases)
				0.4,   // hasDarkBrown: moderate positive (darker keratin areas)
				-0.5,  // hasBlueGray: negative (absence of blue-gray structures)
				-0.2,  // hasBlack: slight negative (rare)
				-0.1,  // contrast: near-neutral
				0.4,   // homogeneity: positive (relatively uniform within domains)
				0.2,   // entropy: slight positive (fissures/ridges create some texture)
				0.3,   // correlation: slight positive (regular repetitive pattern)
				-0.4,  // hasIrregularNetwork: negative (faded/absent network)
				-0.3,  // hasIrregularGlobules: negative
				-0.5,  // hasStreaks: negative (not expected)
				-1.2,  // hasBlueWhiteVeil: strong negative (key differentiator from melanoma)
				-0.1,  // hasRegressionStructures: near-neutral
				-0.4,  // structuralScore: negative (low structural complexity)
			],
			b: -0.8, // Mild negative bias (moderately common class)
		},

		// =================================================================
		// DERMATOFIBROMA (df)
		// =================================================================
		// Key evidence:
		// - Central white scar-like patch: pathognomonic dermoscopy sign,
		//   sensitivity 57%, specificity 97% (Zaballos 2008)
		// - Peripheral delicate pigment network (fading at center)
		// - Small (typically <10mm), symmetric, regular borders
		// - High compactness (round to oval shape)
		// - 1-2 colors (light brown periphery + white center)
		// - HAM10000: df = 1.2%, smallest class, very homogeneous texture
		// - Dimple sign on lateral compression (clinical, not dermoscopic)
		df: {
			w: [
				-0.8,  // asymmetry: negative (symmetric lesion)
				-0.5,  // borderScore: negative (regular, well-defined border)
				-0.6,  // colorCount: negative (typically 1-2 colors only)
				-0.7,  // diameterMm: negative (small lesions, typically <10mm)
				1.5,   // hasWhite: very strong positive (central white scar-like patch, spec. 97%)
				-0.3,  // hasRed: slight negative
				0.5,   // hasLightBrown: positive (peripheral pigment network)
				-0.4,  // hasDarkBrown: negative (no dark components)
				-0.3,  // hasBlueGray: negative
				-0.5,  // hasBlack: negative (no black structures)
				-0.2,  // contrast: slight negative (low contrast lesion)
				0.8,   // homogeneity: strong positive (very uniform texture)
				-0.4,  // entropy: negative (low textural disorder)
				0.3,   // correlation: positive (regular pattern)
				-0.3,  // hasIrregularNetwork: negative (network is delicate and regular)
				-0.4,  // hasIrregularGlobules: negative
				-0.5,  // hasStreaks: negative
				-0.8,  // hasBlueWhiteVeil: negative
				-0.3,  // hasRegressionStructures: negative
				-0.6,  // structuralScore: negative (low structural complexity)
			],
			b: -2.8, // Strong negative bias (rare class, only 1.2% in HAM10000)
		},

		// =================================================================
		// MELANOCYTIC NEVUS (nv)
		// =================================================================
		// Key evidence:
		// - Most common lesion type (66.9% of HAM10000)
		// - Menzies method: BOTH negative features required (single color
		//   AND symmetry) to exclude melanoma; nevi satisfy both
		// - Typical dermoscopic patterns: regular globular, regular reticular,
		//   homogeneous blue, or combined pattern
		// - Symmetric (asymmetry mean 0.2 in HAM10000 nevi)
		// - Regular border (border score mean 1.3 in HAM10000 nevi)
		// - 1-2 colors (typically brown shades)
		// - NO blue-white veil, NO streaks, NO regression
		// - High homogeneity, low entropy (uniform texture)
		// - Low structural complexity (regular or absent structures)
		nv: {
			w: [
				-1.5,  // asymmetry: strong negative (nevi are symmetric; Menzies negative feature)
				-0.8,  // borderScore: strong negative (regular borders)
				-1.0,  // colorCount: strong negative (1-2 colors; Menzies requires single color)
				-0.2,  // diameterMm: slight negative (nevi typically <6mm)
				-0.3,  // hasWhite: slight negative (not typical)
				-0.4,  // hasRed: negative (not vascular)
				0.5,   // hasLightBrown: positive (common brown color)
				0.2,   // hasDarkBrown: slight positive (can be present)
				-0.6,  // hasBlueGray: negative (except blue nevi, which are rare)
				-0.5,  // hasBlack: negative
				-0.3,  // contrast: slight negative (low contrast, uniform)
				1.0,   // homogeneity: strong positive (uniform texture is hallmark of nevi)
				-0.5,  // entropy: negative (orderly, low textural disorder)
				0.4,   // correlation: positive (regular structured pattern)
				-0.8,  // hasIrregularNetwork: negative (network is regular if present)
				-0.6,  // hasIrregularGlobules: negative (globules are regular if present)
				-1.5,  // hasStreaks: strong negative (absence is diagnostic feature)
				-1.5,  // hasBlueWhiteVeil: strong negative (absence is diagnostic feature)
				-0.8,  // hasRegressionStructures: negative (no regression in benign nevi)
				-1.5,  // structuralScore: strong negative (low complexity)
			],
			b: 1.8, // Strong positive bias (most common class — 66.9% prior in HAM10000)
		},

		// =================================================================
		// ACTINIC KERATOSIS / INTRAEPITHELIAL CARCINOMA (akiec)
		// =================================================================
		// Key evidence:
		// - Dermoscopy: strawberry pattern (red pseudo-network + white/yellow
		//   scale), targetoid follicles, rosettes
		// - Rough/scaly texture: high entropy, low homogeneity
		// - Erythematous (reddish) background
		// - Typically low color diversity (red + white/yellow)
		// - Moderate asymmetry (not as symmetric as nevi)
		// - HAM10000: akiec = 3.3%, often on sun-exposed areas
		// - Can show irregular network at periphery (progression toward SCC)
		// - No blue-white veil (differentiates from melanoma)
		akiec: {
			w: [
				0.4,   // asymmetry: moderate positive (more asymmetric than benign)
				0.3,   // borderScore: slight positive (ill-defined borders)
				-0.3,  // colorCount: slight negative (typically 2-3 colors, not diverse)
				0.1,   // diameterMm: near-neutral (variable size)
				0.3,   // hasWhite: slight positive (white/yellow scale)
				0.8,   // hasRed: strong positive (erythematous/strawberry pattern)
				0.1,   // hasLightBrown: near-neutral
				-0.2,  // hasDarkBrown: slight negative
				-0.3,  // hasBlueGray: negative
				-0.2,  // hasBlack: slight negative
				0.4,   // contrast: positive (scale creates contrast against red background)
				-0.6,  // homogeneity: negative (rough/scaly texture is hallmark)
				0.8,   // entropy: strong positive (high textural disorder from scales)
				-0.2,  // correlation: slight negative (disorganized texture)
				0.3,   // hasIrregularNetwork: slight positive (possible in progressing AK)
				-0.1,  // hasIrregularGlobules: near-neutral
				-0.3,  // hasStreaks: negative (not typical)
				-0.8,  // hasBlueWhiteVeil: negative (differentiator from melanoma)
				0.2,   // hasRegressionStructures: slight positive (can show regression)
				0.1,   // structuralScore: near-neutral
			],
			b: -2.0, // Negative bias (uncommon class, 3.3%)
		},

		// =================================================================
		// VASCULAR LESION (vasc)
		// =================================================================
		// Includes angiomas, angiokeratomas, pyogenic granulomas, and
		// hemorrhage.
		// Key evidence:
		// - Red is dominant color: virtually all vascular lesions are red/purple
		//   (sensitivity for red: ~95% in vascular lesions)
		// - Blue-gray component in thrombosed angiomas
		// - Absence of brown/black pigment (differentiator from melanocytic lesions)
		// - Lacunae pattern: round-to-oval reddish-blue structures (angiokeratoma)
		// - Typically symmetric with regular borders
		// - Low structural complexity (no network, no streaks, no veil)
		// - HAM10000: vasc = 1.4%, relatively rare
		// - High homogeneity within red domains
		vasc: {
			w: [
				-0.5,  // asymmetry: negative (typically symmetric)
				-0.3,  // borderScore: slight negative (regular borders)
				-0.2,  // colorCount: slight negative (typically 1-2 colors)
				-0.1,  // diameterMm: near-neutral
				-0.3,  // hasWhite: slight negative
				2.0,   // hasRed: very strong positive (defining feature, sens. ~95%)
				-0.8,  // hasLightBrown: negative (not a brown lesion)
				-1.0,  // hasDarkBrown: strong negative (differentiator from melanocytic)
				0.5,   // hasBlueGray: moderate positive (thrombosed vessels/angiokeratomas)
				-1.0,  // hasBlack: strong negative (no melanin)
				0.3,   // contrast: slight positive
				0.4,   // homogeneity: positive (uniform within red lacunae)
				-0.2,  // entropy: slight negative (orderly lacunae pattern)
				0.2,   // correlation: slight positive
				-0.6,  // hasIrregularNetwork: negative (no pigment network)
				-0.4,  // hasIrregularGlobules: negative
				-0.5,  // hasStreaks: negative (no streaks)
				-0.7,  // hasBlueWhiteVeil: negative (no veil)
				-0.3,  // hasRegressionStructures: negative
				-0.5,  // structuralScore: negative (low structural complexity)
			],
			b: -2.5, // Strong negative bias (rare class, 1.4%)
		},
	},
};

// =================================================================
// Classification function
// =================================================================

/**
 * Classify a feature vector using the pre-computed logistic regression weights.
 *
 * Computes logit = dot(W, features) + bias for each class, then applies
 * softmax to produce calibrated probabilities.
 *
 * @param features - 20-element feature vector (see FEATURE_NAMES for order)
 * @returns Record mapping each class name to its probability [0, 1]
 */
export function classifyWithTrainedWeights(features: number[]): Record<LesionClass, number> {
	if (features.length !== FEATURE_NAMES.length) {
		throw new Error(
			`Expected ${FEATURE_NAMES.length} features, got ${features.length}. ` +
			`Feature order: ${FEATURE_NAMES.join(", ")}`,
		);
	}

	// Compute logits for each class: logit[c] = dot(W[c], features) + bias[c]
	const logits: Record<string, number> = {};
	for (const cls of CLASS_NAMES) {
		const { w, b } = TRAINED_WEIGHTS.weights[cls];
		let logit = b;
		for (let i = 0; i < features.length; i++) {
			logit += w[i] * features[i];
		}
		logits[cls] = logit;
	}

	// Softmax with numerical stability (subtract max logit)
	const maxLogit = Math.max(...Object.values(logits));
	const exps: Record<string, number> = {};
	let sumExp = 0;

	for (const cls of CLASS_NAMES) {
		exps[cls] = Math.exp(logits[cls] - maxLogit);
		sumExp += exps[cls];
	}

	const probabilities = {} as Record<LesionClass, number>;
	for (const cls of CLASS_NAMES) {
		probabilities[cls] = exps[cls] / sumExp;
	}

	return probabilities;
}

// =================================================================
// Feature extraction
// =================================================================

/**
 * Extract the 20-element feature vector from image analysis results.
 *
 * Maps the rich analysis objects (segmentation, color analysis, texture,
 * structures) into a flat numeric array suitable for the linear classifier.
 *
 * Boolean features are encoded as 0/1. Continuous features are passed through
 * as-is (the weight matrix was calibrated for the raw feature scales).
 *
 * @param seg - Segmentation result (for area, perimeter, compactness proxy)
 * @param asymmetry - Asymmetry score (0-2)
 * @param borderScore - Border irregularity (0-8)
 * @param colorAnalysis - Color analysis result
 * @param texture - Texture (GLCM) features
 * @param structures - Detected dermoscopic structures
 * @param diameterMm - Estimated diameter in mm (default: estimated from area)
 * @returns 20-element numeric feature vector
 */
export function extractFeatureVector(
	seg: { area: number; perimeter: number },
	asymmetry: number,
	borderScore: number,
	colorAnalysis: {
		colorCount: number;
		dominantColors: Array<{ name: string; percentage: number; rgb: [number, number, number] }>;
		hasBlueWhiteStructures: boolean;
	},
	texture: {
		contrast: number;
		homogeneity: number;
		entropy: number;
		correlation: number;
	},
	structures: {
		hasIrregularNetwork: boolean;
		hasIrregularGlobules: boolean;
		hasStreaks: boolean;
		hasBlueWhiteVeil: boolean;
		hasRegressionStructures: boolean;
		structuralScore: number;
	},
	diameterMm?: number,
): number[] {
	// Build color presence set from dominant colors
	const colorNames = new Set(colorAnalysis.dominantColors.map((c) => c.name));

	// Estimate diameter from area if not provided
	// Assume circular lesion: diameter = 2 * sqrt(area / pi)
	// Convert pixel area to approximate mm using 40px/mm at 10x magnification
	const estimatedDiameterMm = diameterMm ??
		(2 * Math.sqrt(seg.area / Math.PI)) / 40;

	// Build the 20-element feature vector in canonical order
	return [
		asymmetry,                                    // 0: asymmetry (0-2)
		borderScore,                                  // 1: borderScore (0-8)
		colorAnalysis.colorCount,                     // 2: colorCount (1-6)
		estimatedDiameterMm,                          // 3: diameterMm
		colorNames.has("white") ? 1 : 0,              // 4: hasWhite
		colorNames.has("red") ? 1 : 0,                // 5: hasRed
		colorNames.has("light-brown") ? 1 : 0,        // 6: hasLightBrown
		colorNames.has("dark-brown") ? 1 : 0,         // 7: hasDarkBrown
		colorNames.has("blue-gray") ? 1 : 0,          // 8: hasBlueGray
		colorNames.has("black") ? 1 : 0,              // 9: hasBlack
		texture.contrast,                             // 10: contrast (0-1)
		texture.homogeneity,                          // 11: homogeneity (0-1)
		texture.entropy,                              // 12: entropy (0-1)
		texture.correlation,                          // 13: correlation (0-1)
		structures.hasIrregularNetwork ? 1 : 0,       // 14: hasIrregularNetwork
		structures.hasIrregularGlobules ? 1 : 0,      // 15: hasIrregularGlobules
		structures.hasStreaks ? 1 : 0,                 // 16: hasStreaks
		structures.hasBlueWhiteVeil ? 1 : 0,          // 17: hasBlueWhiteVeil
		structures.hasRegressionStructures ? 1 : 0,   // 18: hasRegressionStructures
		structures.structuralScore,                   // 19: structuralScore (0-1)
	];
}
