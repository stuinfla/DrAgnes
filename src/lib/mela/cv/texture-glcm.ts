/**
 * Texture analysis using Gray-Level Co-occurrence Matrix (GLCM).
 *
 * Computes contrast, homogeneity, entropy, and correlation
 * features from the lesion region.
 */

import type { TextureResult } from "./types";

/**
 * Compute Gray-Level Co-occurrence Matrix (GLCM) texture features
 * within the lesion mask.
 *
 * Features computed:
 * - Contrast: intensity differences between neighboring pixels
 * - Homogeneity: closeness of GLCM elements to the diagonal
 * - Entropy: randomness/complexity of texture
 * - Correlation: linear dependency of gray levels
 *
 * @returns Normalized texture features
 */
export function analyzeTexture(imageData: ImageData, mask: Uint8Array): TextureResult {
	const { data, width, height } = imageData;

	// Quantize grayscale to 32 levels for GLCM (reduces computation)
	const levels = 32;
	const quantized = new Uint8Array(width * height);
	for (let i = 0; i < width * height; i++) {
		if (mask[i] !== 1) continue;
		const px = i * 4;
		const gray = 0.299 * data[px] + 0.587 * data[px + 1] + 0.114 * data[px + 2];
		quantized[i] = Math.min(levels - 1, Math.floor(gray / 256 * levels));
	}

	// Build GLCM for 4 directions: 0, 45, 90, 135 degrees
	const directions: [number, number][] = [[1, 0], [1, 1], [0, 1], [-1, 1]];
	const glcm = new Float64Array(levels * levels);
	let totalPairs = 0;

	for (const [dx, dy] of directions) {
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const idx = y * width + x;
				if (mask[idx] !== 1) continue;

				const nx = x + dx;
				const ny = y + dy;
				if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
				const nIdx = ny * width + nx;
				if (mask[nIdx] !== 1) continue;

				const i = quantized[idx];
				const j = quantized[nIdx];
				glcm[i * levels + j]++;
				glcm[j * levels + i]++; // Symmetric
				totalPairs += 2;
			}
		}
	}

	// Normalize GLCM
	if (totalPairs > 0) {
		for (let i = 0; i < glcm.length; i++) {
			glcm[i] /= totalPairs;
		}
	}

	// Compute marginal means and standard deviations for correlation
	const muI = new Float64Array(levels);
	const muJ = new Float64Array(levels);
	for (let i = 0; i < levels; i++) {
		for (let j = 0; j < levels; j++) {
			const p = glcm[i * levels + j];
			muI[i] += p;
			muJ[j] += p;
		}
	}

	let meanI = 0;
	let meanJ = 0;
	for (let k = 0; k < levels; k++) {
		meanI += k * muI[k];
		meanJ += k * muJ[k];
	}

	let stdI = 0;
	let stdJ = 0;
	for (let k = 0; k < levels; k++) {
		stdI += (k - meanI) ** 2 * muI[k];
		stdJ += (k - meanJ) ** 2 * muJ[k];
	}
	stdI = Math.sqrt(stdI);
	stdJ = Math.sqrt(stdJ);

	// Compute features
	let contrast = 0;
	let homogeneity = 0;
	let entropy = 0;
	let correlation = 0;

	for (let i = 0; i < levels; i++) {
		for (let j = 0; j < levels; j++) {
			const p = glcm[i * levels + j];
			if (p === 0) continue;

			contrast += (i - j) ** 2 * p;
			homogeneity += p / (1 + (i - j) ** 2);
			entropy -= p * Math.log2(p + 1e-10);

			if (stdI > 0 && stdJ > 0) {
				correlation += (i - meanI) * (j - meanJ) * p / (stdI * stdJ);
			}
		}
	}

	// Normalize to [0, 1] ranges for comparability
	const maxContrast = (levels - 1) ** 2;

	return {
		contrast: Math.min(1, contrast / maxContrast),
		homogeneity,
		entropy: entropy / Math.log2(levels * levels), // normalize by max possible entropy
		correlation: (correlation + 1) / 2, // map [-1,1] to [0,1]
	};
}
