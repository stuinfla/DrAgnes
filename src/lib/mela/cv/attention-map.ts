/**
 * Diagnostically-weighted attention heatmap generation.
 *
 * Combines color irregularity, structural complexity (local entropy),
 * and border proximity into a 224x224 attention map.
 */

import { rgbToLab, toGrayscale } from "./color-space";

/**
 * Generate a diagnostically-weighted attention heatmap showing which
 * regions of the image are most relevant for classification.
 *
 * Combines:
 * - Color irregularity (deviation from mean lesion color in LAB)
 * - Structural complexity (local entropy in grayscale)
 * - Border proximity (active edges are diagnostically important)
 *
 * @returns Normalized 224x224 Float32Array with values [0, 1]
 */
export function generateAttentionMap(
	imageData: ImageData,
	mask: Uint8Array,
	width: number,
	height: number,
): Float32Array {
	const { data } = imageData;
	const targetSize = 224;

	// Compute mean lesion color in LAB
	let meanL = 0;
	let meanA = 0;
	let meanB = 0;
	let lesionCount = 0;

	for (let i = 0; i < width * height; i++) {
		if (mask[i] !== 1) continue;
		const px = i * 4;
		const [L, a, b] = rgbToLab(data[px], data[px + 1], data[px + 2]);
		meanL += L;
		meanA += a;
		meanB += b;
		lesionCount++;
	}

	if (lesionCount > 0) {
		meanL /= lesionCount;
		meanA /= lesionCount;
		meanB /= lesionCount;
	}

	// Compute attention components at original resolution
	const attention = new Float32Array(width * height);
	const gray = toGrayscale(imageData);

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const idx = y * width + x;
			if (mask[idx] !== 1) {
				attention[idx] = 0;
				continue;
			}

			const px = idx * 4;

			// 1. Color irregularity (delta E from mean)
			const [L, a, b] = rgbToLab(data[px], data[px + 1], data[px + 2]);
			const dE = Math.sqrt((L - meanL) ** 2 + (a - meanA) ** 2 + (b - meanB) ** 2);
			const colorScore = Math.min(1, dE / 40); // normalize: dE > 40 = max

			// 2. Local entropy (3x3 neighborhood)
			let localEntropy = 0;
			if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
				const localHist = new Float64Array(16); // 4-bit quantized
				let localCount = 0;
				for (let dy = -1; dy <= 1; dy++) {
					for (let dx = -1; dx <= 1; dx++) {
						const nIdx = (y + dy) * width + (x + dx);
						if (mask[nIdx] === 1) {
							const bin = Math.min(15, Math.floor(gray[nIdx] / 16));
							localHist[bin]++;
							localCount++;
						}
					}
				}
				if (localCount > 0) {
					for (let h = 0; h < 16; h++) {
						if (localHist[h] > 0) {
							const p = localHist[h] / localCount;
							localEntropy -= p * Math.log2(p);
						}
					}
					localEntropy /= 4; // normalize by log2(16) = 4
				}
			}

			// 3. Border proximity
			// Quick check: distance to nearest background pixel
			let borderDist = Infinity;
			const searchR = 15;
			for (let dy = -searchR; dy <= searchR && borderDist > 1; dy++) {
				for (let dx = -searchR; dx <= searchR; dx++) {
					const nx = x + dx;
					const ny = y + dy;
					if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
						const d = Math.sqrt(dx * dx + dy * dy);
						if (d < borderDist) borderDist = d;
						continue;
					}
					if (mask[ny * width + nx] === 0) {
						const d = Math.sqrt(dx * dx + dy * dy);
						if (d < borderDist) borderDist = d;
					}
				}
			}
			const borderScore = borderDist < searchR ? 1 - borderDist / searchR : 0;

			// Weighted combination
			attention[idx] = 0.45 * colorScore + 0.30 * localEntropy + 0.25 * borderScore;
		}
	}

	// Gaussian smooth the attention map (sigma=3)
	const smoothed = gaussianSmooth(attention, width, height, 3);

	// Normalize to [0, 1]
	let maxVal = 0;
	for (let i = 0; i < smoothed.length; i++) {
		if (smoothed[i] > maxVal) maxVal = smoothed[i];
	}
	if (maxVal > 0) {
		for (let i = 0; i < smoothed.length; i++) {
			smoothed[i] /= maxVal;
		}
	}

	// Resize to 224x224
	return resizeFloat32(smoothed, width, height, targetSize, targetSize);
}

/**
 * Gaussian smoothing with separable 1D convolution.
 */
export function gaussianSmooth(data: Float32Array, w: number, h: number, sigma: number): Float32Array {
	const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
	const halfK = Math.floor(kernelSize / 2);
	const kernel = new Float64Array(kernelSize);
	let kernelSum = 0;

	for (let i = 0; i < kernelSize; i++) {
		const x = i - halfK;
		kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
		kernelSum += kernel[i];
	}
	for (let i = 0; i < kernelSize; i++) kernel[i] /= kernelSum;

	// Horizontal pass
	const hPass = new Float32Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			let sum = 0;
			let weightSum = 0;
			for (let k = -halfK; k <= halfK; k++) {
				const nx = x + k;
				if (nx >= 0 && nx < w) {
					const kw = kernel[k + halfK];
					sum += data[y * w + nx] * kw;
					weightSum += kw;
				}
			}
			hPass[y * w + x] = weightSum > 0 ? sum / weightSum : 0;
		}
	}

	// Vertical pass
	const result = new Float32Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			let sum = 0;
			let weightSum = 0;
			for (let k = -halfK; k <= halfK; k++) {
				const ny = y + k;
				if (ny >= 0 && ny < h) {
					const kw = kernel[k + halfK];
					sum += hPass[ny * w + x] * kw;
					weightSum += kw;
				}
			}
			result[y * w + x] = weightSum > 0 ? sum / weightSum : 0;
		}
	}

	return result;
}

/**
 * Bilinear resize for Float32Array (single-channel).
 */
export function resizeFloat32(
	src: Float32Array,
	srcW: number,
	srcH: number,
	dstW: number,
	dstH: number,
): Float32Array {
	const result = new Float32Array(dstW * dstH);
	const xRatio = srcW / dstW;
	const yRatio = srcH / dstH;

	for (let y = 0; y < dstH; y++) {
		for (let x = 0; x < dstW; x++) {
			const srcX = x * xRatio;
			const srcY = y * yRatio;
			const x0 = Math.floor(srcX);
			const y0 = Math.floor(srcY);
			const x1 = Math.min(x0 + 1, srcW - 1);
			const y1 = Math.min(y0 + 1, srcH - 1);
			const dx = srcX - x0;
			const dy = srcY - y0;

			const topLeft = src[y0 * srcW + x0];
			const topRight = src[y0 * srcW + x1];
			const botLeft = src[y1 * srcW + x0];
			const botRight = src[y1 * srcW + x1];

			const top = topLeft + (topRight - topLeft) * dx;
			const bot = botLeft + (botRight - botLeft) * dx;
			result[y * dstW + x] = top + (bot - top) * dy;
		}
	}

	return result;
}
