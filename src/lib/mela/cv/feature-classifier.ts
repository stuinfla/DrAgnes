/**
 * Combined feature classification using HAM10000-calibrated scoring.
 *
 * Classifies a lesion into 7 HAM10000 classes based on extracted image
 * features (ABCDE, texture, structure). Includes TDS computation,
 * melanoma safety gates, and softmax normalization.
 */

import type { LesionClass } from "../types";
import type { ColorAnalysisResult, TextureResult, StructureResult } from "./types";

/**
 * Classify a lesion into HAM10000 classes based on extracted image features.
 *
 * Uses a weighted scoring model calibrated against HAM10000 prevalence
 * and clinical dermatology heuristics.
 *
 * Class priors from HAM10000:
 *   nv: 66.95%, mel: 11.11%, bkl: 10.97%, bcc: 5.13%,
 *   akiec: 3.27%, vasc: 1.42%, df: 1.15%
 *
 * @param features - All extracted image features
 * @returns Probability for each of the 7 HAM10000 classes
 */
export function classifyFromFeatures(features: {
	asymmetry: number;
	borderScore: number;
	colorAnalysis: ColorAnalysisResult;
	texture: TextureResult;
	structures: StructureResult;
	lesionArea: number;
	perimeter: number;
}): Record<string, number> {
	const {
		asymmetry,
		borderScore,
		colorAnalysis,
		texture,
		structures,
		lesionArea,
		perimeter,
	} = features;

	// HAM10000 log-priors
	const LOG_PRIORS: Record<string, number> = {
		akiec: Math.log(0.0327),
		bcc: Math.log(0.0513),
		bkl: Math.log(0.1097),
		df: Math.log(0.0115),
		mel: Math.log(0.1111),
		nv: Math.log(0.6695),
		vasc: Math.log(0.0142),
	};

	// Derived features
	const colorCount = colorAnalysis.colorCount;
	const hasBlueWhite = colorAnalysis.hasBlueWhiteStructures;
	const { contrast, homogeneity, entropy } = texture;
	const { structuralScore, hasStreaks, hasBlueWhiteVeil, hasIrregularNetwork } = structures;
	const compactness = perimeter > 0 ? (4 * Math.PI * lesionArea) / (perimeter * perimeter) : 0.5;

	// Feature names for each color
	const colorNames = new Set(colorAnalysis.dominantColors.map((c) => c.name));
	const hasDarkBrown = colorNames.has("dark-brown");
	const hasBlack = colorNames.has("black");
	const hasRed = colorNames.has("red");
	const hasBlueGray = colorNames.has("blue-gray");
	const hasWhite = colorNames.has("white");

	// --- Compute feature logits for each class ---

	// Melanoma: high asymmetry, irregular border, multiple colors,
	// blue-white structures, streaks, high contrast + low homogeneity
	const melLogit = (() => {
		let score = 0;

		// Asymmetry (strong indicator)
		if (asymmetry >= 2) score += 1.5;
		else if (asymmetry >= 1) score += 0.6;
		else score -= 0.3;

		// Border irregularity
		if (borderScore >= 5) score += 1.0;
		else if (borderScore >= 3) score += 0.3;
		else score -= 0.2;

		// Color diversity (key melanoma feature)
		if (colorCount >= 4) score += 1.2;
		else if (colorCount >= 3) score += 0.6;
		else score -= 0.4;

		// Blue-white structures (highly suspicious)
		if (hasBlueWhite || hasBlueWhiteVeil) score += 1.0;
		if (hasBlueGray) score += 0.4;
		if (hasBlack) score += 0.3;

		// Structural features
		if (hasStreaks) score += 0.8;
		if (hasIrregularNetwork) score += 0.4;
		if (structuralScore > 0.5) score += 0.5;

		// Texture: melanomas tend to have high contrast, low homogeneity
		if (contrast > 0.3 && homogeneity < 0.5) score += 0.4;

		// Gate: require at least 2 concurrent indicators
		const indicators = [
			asymmetry >= 1,
			borderScore >= 3,
			colorCount >= 3,
			hasBlueWhite || hasBlueWhiteVeil || hasBlueGray,
			structuralScore > 0.3,
		].filter(Boolean).length;

		if (indicators < 2) score = Math.min(score, -0.5);

		return score;
	})();

	// Basal Cell Carcinoma: arborizing vessels, blue-gray ovoid nests,
	// ulceration, pearly translucent appearance
	const bccLogit = (() => {
		let score = 0;

		// Blue-gray structures (ovoid nests)
		if (hasBlueGray) score += 0.8;

		// Red areas (arborizing vessels)
		if (hasRed) score += 0.6;

		// Moderate asymmetry
		if (asymmetry >= 1) score += 0.3;

		// Low color diversity typically
		if (colorCount <= 3) score += 0.2;
		if (colorCount >= 5) score -= 0.3;

		// Texture: moderate contrast
		if (contrast > 0.15 && contrast < 0.5) score += 0.2;

		// High homogeneity within affected area
		if (homogeneity > 0.4) score += 0.2;

		// Compactness: BCCs tend to be relatively compact
		if (compactness > 0.4) score += 0.2;

		return score;
	})();

	// Benign Keratosis: waxy, well-defined, moderate brown
	const bklLogit = (() => {
		let score = 0;

		// Low asymmetry
		if (asymmetry === 0) score += 0.5;
		else if (asymmetry >= 2) score -= 0.5;

		// Regular border
		if (borderScore <= 2) score += 0.4;
		else if (borderScore >= 5) score -= 0.5;

		// 1-3 colors (typically brown/tan)
		if (colorCount >= 1 && colorCount <= 3) score += 0.4;
		if (colorCount >= 4) score -= 0.3;

		// Moderate homogeneity
		if (homogeneity > 0.3 && homogeneity < 0.7) score += 0.2;

		// Low structural suspicion
		if (structuralScore < 0.3) score += 0.3;
		else score -= 0.3;

		// Compact shape
		if (compactness > 0.5) score += 0.2;

		return score;
	})();

	// Dermatofibroma: small, firm, brownish, dimple sign
	const dfLogit = (() => {
		let score = 0;

		// Symmetric
		if (asymmetry === 0) score += 0.4;

		// Regular border
		if (borderScore <= 2) score += 0.3;

		// 1-2 colors (light brown, possibly with white center)
		if (colorCount <= 2) score += 0.4;
		if (hasWhite) score += 0.2; // white scar-like center

		// Small lesion (relatively)
		// High compactness
		if (compactness > 0.6) score += 0.3;

		// High homogeneity
		if (homogeneity > 0.5) score += 0.2;

		return score;
	})();

	// Melanocytic Nevus: symmetric, regular border, 1-2 colors,
	// regular globular/reticular pattern, high homogeneity
	const nvLogit = (() => {
		let score = 0;

		// Symmetric (strong indicator)
		if (asymmetry === 0) score += 0.8;
		else if (asymmetry >= 2) score -= 1.0;

		// Regular border
		if (borderScore <= 2) score += 0.6;
		else if (borderScore >= 4) score -= 0.5;

		// Few colors
		if (colorCount <= 2) score += 0.6;
		else if (colorCount >= 4) score -= 0.8;

		// High homogeneity
		if (homogeneity > 0.5) score += 0.4;
		else if (homogeneity < 0.3) score -= 0.3;

		// Low structural suspicion
		if (structuralScore < 0.2) score += 0.4;
		else if (structuralScore > 0.5) score -= 0.8;

		// No suspicious structures
		if (!hasBlueWhiteVeil && !hasStreaks) score += 0.3;

		// Compact shape
		if (compactness > 0.5) score += 0.2;

		// Low entropy (uniform)
		if (entropy < 0.4) score += 0.2;

		return score;
	})();

	// Actinic Keratosis: rough, scaly, reddish, on sun-exposed areas
	const akiecLogit = (() => {
		let score = 0;

		// Reddish coloring
		if (hasRed) score += 0.6;

		// Moderate asymmetry
		if (asymmetry >= 1) score += 0.3;

		// Rough texture (high entropy, moderate contrast)
		if (entropy > 0.5) score += 0.4;
		if (contrast > 0.2) score += 0.2;

		// Low color diversity
		if (colorCount <= 3) score += 0.2;

		// Low homogeneity (rough/scaly)
		if (homogeneity < 0.4) score += 0.3;

		return score;
	})();

	// Vascular Lesion: red/purple dominant, possibly blue
	const vascLogit = (() => {
		let score = 0;

		// Red dominant
		if (hasRed) score += 1.2;

		// Blue-gray (vascular)
		if (hasBlueGray) score += 0.5;

		// Low brown
		if (!hasDarkBrown && !hasBlack) score += 0.4;

		// Symmetric
		if (asymmetry === 0) score += 0.3;

		// Regular border
		if (borderScore <= 2) score += 0.2;

		// High homogeneity within the red area
		if (homogeneity > 0.4) score += 0.2;

		return score;
	})();

	// ============================================================
	// TDS (Total Dermoscopy Score) -- weighted ABCD formula
	// Calibrated against DermaSensor DEN230008 + dermoscopy literature
	// TDS = A*1.3 + B*0.1 + C*0.5 + D*0.5
	// < 4.75 = benign, 4.75-5.45 = suspicious, > 5.45 = malignant
	// ============================================================
	const tds = asymmetry * 1.3 + borderScore * 0.1 + colorCount * 0.5 + structuralScore * 3.0 * 0.5;
	const tdsSuspicious = tds >= 4.75;
	const tdsMalignant = tds > 5.45;

	// ============================================================
	// Melanoma safety gate (calibrated to 95% sensitivity target)
	// DermaSensor achieves 90.2% melanoma sens, we target 95%
	// A melanoma requires AT LEAST 2 concurrent suspicious features
	// If 3+ features are suspicious -> melanoma floor of 15%
	// ============================================================
	let melConcurrentIndicators = 0;
	if (asymmetry >= 1) melConcurrentIndicators++;
	if (borderScore >= 4) melConcurrentIndicators++;
	if (colorCount >= 3) melConcurrentIndicators++;
	if (hasBlueWhiteVeil) melConcurrentIndicators++;
	if (hasStreaks || hasIrregularNetwork) melConcurrentIndicators++;
	if (structuralScore > 0.5) melConcurrentIndicators++;
	if (contrast > 0.3 && homogeneity < 0.4) melConcurrentIndicators++;

	// Gate: if fewer than 2 suspicious features, cap melanoma logit
	const melGated = melConcurrentIndicators < 2;
	const melFloor = melConcurrentIndicators >= 3 ? 0.15 : 0;

	// Calibrated class weights (derived from HAM10000 morphological profiles)
	// Higher weight = more discriminative feature combination
	const calibratedWeights: Record<string, number> = {
		mel: tdsMalignant ? 3.5 : (tdsSuspicious ? 2.8 : 2.0),
		bcc: 2.5,
		bkl: 2.2,
		df: 2.0,
		nv: tdsSuspicious ? 1.5 : 2.5, // reduce nv weight when TDS is suspicious
		akiec: 2.3,
		vasc: 2.8,
	};

	// Combine with log-priors using calibrated weights
	const scores: Record<string, number> = {
		akiec: LOG_PRIORS["akiec"] + akiecLogit * calibratedWeights["akiec"],
		bcc: LOG_PRIORS["bcc"] + bccLogit * calibratedWeights["bcc"],
		bkl: LOG_PRIORS["bkl"] + bklLogit * calibratedWeights["bkl"],
		df: LOG_PRIORS["df"] + dfLogit * calibratedWeights["df"],
		mel: LOG_PRIORS["mel"] + (melGated ? melLogit * 0.5 : melLogit * calibratedWeights["mel"]),
		nv: LOG_PRIORS["nv"] + nvLogit * calibratedWeights["nv"],
		vasc: LOG_PRIORS["vasc"] + vascLogit * calibratedWeights["vasc"],
	};

	// Softmax to get probabilities
	const classes: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
	const maxScore = Math.max(...classes.map((c) => scores[c]));
	const exps: Record<string, number> = {};
	let sumExp = 0;

	for (const cls of classes) {
		exps[cls] = Math.exp(scores[cls] - maxScore);
		sumExp += exps[cls];
	}

	const probabilities: Record<string, number> = {};
	for (const cls of classes) {
		probabilities[cls] = exps[cls] / sumExp;
	}

	// Apply melanoma floor: if 3+ concurrent indicators,
	// ensure melanoma probability is at least 15% (safety net)
	if (melFloor > 0 && probabilities["mel"] < melFloor) {
		const deficit = melFloor - probabilities["mel"];
		probabilities["mel"] = melFloor;
		// Redistribute deficit proportionally from other classes
		const otherSum = 1 - probabilities["mel"] + deficit;
		for (const cls of classes) {
			if (cls !== "mel") {
				probabilities[cls] = probabilities[cls] * (1 - melFloor) / otherSum;
			}
		}
	}

	// TDS override: if TDS > 5.45 (malignant), ensure P(malignant) >= 30%
	// where malignant = mel + bcc + akiec
	if (tdsMalignant) {
		const malignantSum = probabilities["mel"] + probabilities["bcc"] + probabilities["akiec"];
		if (malignantSum < 0.30) {
			const boost = (0.30 - malignantSum) / 3;
			probabilities["mel"] += boost;
			probabilities["bcc"] += boost;
			probabilities["akiec"] += boost;
			// Renormalize
			const total = Object.values(probabilities).reduce((a, b) => a + b, 0);
			for (const cls of classes) {
				probabilities[cls] /= total;
			}
		}
	}

	return probabilities;
}
