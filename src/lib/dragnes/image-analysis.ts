/**
 * DrAgnes Dermoscopic Image Analysis Engine
 *
 * Performs real computer vision analysis on skin lesion images
 * for classification and ABCDE scoring. All processing uses pure
 * TypeScript with typed arrays -- no external dependencies.
 *
 * Pipeline:
 *   1. Lesion segmentation (LAB color space + Otsu + connected components)
 *   2. Asymmetry measurement (principal axis folding)
 *   3. Border analysis (8-octant irregularity)
 *   4. Color analysis (k-means clustering of dermoscopic colors)
 *   5. Texture analysis (GLCM features)
 *   6. Structural pattern detection (LBP + frequency analysis)
 *   7. Attention heatmap generation
 *   8. Combined feature classification
 */

import type { LesionClass } from "./types";

// ============================================================
// Types
// ============================================================

export interface BBox {
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface SegmentationResult {
	mask: Uint8Array;
	bbox: BBox;
	area: number;
	perimeter: number;
}

export interface ColorAnalysisResult {
	colorCount: number;
	dominantColors: Array<{ name: string; percentage: number; rgb: [number, number, number] }>;
	hasBlueWhiteStructures: boolean;
}

export interface TextureResult {
	contrast: number;
	homogeneity: number;
	entropy: number;
	correlation: number;
}

export interface StructureResult {
	hasIrregularNetwork: boolean;
	hasIrregularGlobules: boolean;
	hasStreaks: boolean;
	hasBlueWhiteVeil: boolean;
	hasRegressionStructures: boolean;
	structuralScore: number;
}

// ============================================================
// Helper: RGB to LAB conversion
// ============================================================

function rgbToLab(r: number, g: number, b: number): [number, number, number] {
	// Normalize to [0,1]
	let rn = r / 255;
	let gn = g / 255;
	let bn = b / 255;

	// sRGB to linear
	rn = rn > 0.04045 ? Math.pow((rn + 0.055) / 1.055, 2.4) : rn / 12.92;
	gn = gn > 0.04045 ? Math.pow((gn + 0.055) / 1.055, 2.4) : gn / 12.92;
	bn = bn > 0.04045 ? Math.pow((bn + 0.055) / 1.055, 2.4) : bn / 12.92;

	// Linear RGB to XYZ (D65 illuminant)
	let x = (rn * 0.4124564 + gn * 0.3575761 + bn * 0.1804375) / 0.95047;
	let y = (rn * 0.2126729 + gn * 0.7151522 + bn * 0.0721750);
	let z = (rn * 0.0193339 + gn * 0.1191920 + bn * 0.9503041) / 1.08883;

	// XYZ to LAB
	const epsilon = 0.008856;
	const kappa = 903.3;
	x = x > epsilon ? Math.cbrt(x) : (kappa * x + 16) / 116;
	y = y > epsilon ? Math.cbrt(y) : (kappa * y + 16) / 116;
	z = z > epsilon ? Math.cbrt(z) : (kappa * z + 16) / 116;

	const L = 116 * y - 16;
	const a = 500 * (x - y);
	const bLab = 200 * (y - z);

	return [L, a, bLab];
}

// ============================================================
// Helper: Grayscale conversion
// ============================================================

function toGrayscale(imageData: ImageData): Uint8Array {
	const { data, width, height } = imageData;
	const gray = new Uint8Array(width * height);
	for (let i = 0; i < gray.length; i++) {
		const px = i * 4;
		gray[i] = Math.round(0.299 * data[px] + 0.587 * data[px + 1] + 0.114 * data[px + 2]);
	}
	return gray;
}

// ============================================================
// Helper: Otsu threshold
// ============================================================

function otsuThreshold(values: Uint8Array | Float32Array, count: number): number {
	const histogram = new Int32Array(256);
	for (let i = 0; i < count; i++) {
		const v = Math.max(0, Math.min(255, Math.round(Number(values[i]))));
		histogram[v]++;
	}

	let sumAll = 0;
	for (let i = 0; i < 256; i++) sumAll += i * histogram[i];

	let sumBg = 0;
	let weightBg = 0;
	let maxVariance = 0;
	let bestThreshold = 0;

	for (let t = 0; t < 256; t++) {
		weightBg += histogram[t];
		if (weightBg === 0) continue;
		const weightFg = count - weightBg;
		if (weightFg === 0) break;

		sumBg += t * histogram[t];
		const meanBg = sumBg / weightBg;
		const meanFg = (sumAll - sumBg) / weightFg;
		const variance = weightBg * weightFg * (meanBg - meanFg) ** 2;

		if (variance > maxVariance) {
			maxVariance = variance;
			bestThreshold = t;
		}
	}

	return bestThreshold;
}

// ============================================================
// Helper: Find largest connected component
// ============================================================

function largestConnectedComponent(mask: Uint8Array, width: number, height: number): Uint8Array {
	const labels = new Int32Array(width * height);
	let nextLabel = 1;
	const labelSizes = new Map<number, number>();

	// BFS flood fill to label connected components
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const idx = y * width + x;
			if (mask[idx] !== 1 || labels[idx] !== 0) continue;

			// BFS from this pixel
			const label = nextLabel++;
			let size = 0;
			const queue: number[] = [idx];
			labels[idx] = label;

			while (queue.length > 0) {
				const cur = queue.pop()!;
				size++;
				const cy = Math.floor(cur / width);
				const cx = cur % width;

				for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
					const nx = cx + dx;
					const ny = cy + dy;
					if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
					const ni = ny * width + nx;
					if (mask[ni] === 1 && labels[ni] === 0) {
						labels[ni] = label;
						queue.push(ni);
					}
				}
			}

			labelSizes.set(label, size);
		}
	}

	// Find the largest component
	let largestLabel = 0;
	let largestSize = 0;
	for (const [label, size] of labelSizes) {
		if (size > largestSize) {
			largestSize = size;
			largestLabel = label;
		}
	}

	// Build output mask with only the largest component
	const result = new Uint8Array(width * height);
	for (let i = 0; i < labels.length; i++) {
		result[i] = labels[i] === largestLabel ? 1 : 0;
	}

	return result;
}

// ============================================================
// Helper: Morphological operations
// ============================================================

function morphDilate(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
	const out = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			let val = 0;
			for (let dy = -r; dy <= r && !val; dy++) {
				for (let dx = -r; dx <= r && !val; dx++) {
					const ny = y + dy;
					const nx = x + dx;
					if (ny >= 0 && ny < h && nx >= 0 && nx < w && mask[ny * w + nx] === 1) {
						val = 1;
					}
				}
			}
			out[y * w + x] = val;
		}
	}
	return out;
}

function morphErode(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
	const out = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			let val = 1;
			for (let dy = -r; dy <= r && val; dy++) {
				for (let dx = -r; dx <= r && val; dx++) {
					const ny = y + dy;
					const nx = x + dx;
					if (ny < 0 || ny >= h || nx < 0 || nx >= w || mask[ny * w + nx] === 0) {
						val = 0;
					}
				}
			}
			out[y * w + x] = val;
		}
	}
	return out;
}

function morphClose(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
	return morphErode(morphDilate(mask, w, h, r), w, h, r);
}

function morphOpen(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
	return morphDilate(morphErode(mask, w, h, r), w, h, r);
}

// ============================================================
// 1. LESION SEGMENTATION
// ============================================================

/**
 * Segment the lesion from surrounding skin.
 *
 * Uses LAB color space L-channel with Otsu thresholding,
 * morphological cleanup, and largest connected component extraction.
 *
 * @param imageData - RGBA ImageData from canvas
 * @returns Binary mask, bounding box, area, and perimeter
 */
export function segmentLesion(imageData: ImageData): SegmentationResult {
	const { data, width, height } = imageData;
	const totalPixels = width * height;

	// Convert to LAB color space, extract L channel (better for skin segmentation)
	const lChannel = new Uint8Array(totalPixels);
	for (let i = 0; i < totalPixels; i++) {
		const px = i * 4;
		const [L] = rgbToLab(data[px], data[px + 1], data[px + 2]);
		// Scale L from [0,100] to [0,255]
		lChannel[i] = Math.max(0, Math.min(255, Math.round(L * 2.55)));
	}

	// Otsu threshold on L channel
	const threshold = otsuThreshold(lChannel, totalPixels);

	// Binary mask: lesion is darker (lower L) than background
	const initialMask = new Uint8Array(totalPixels);
	for (let i = 0; i < totalPixels; i++) {
		initialMask[i] = lChannel[i] <= threshold ? 1 : 0;
	}

	// Morphological cleanup: close small gaps then open to remove noise
	const closedMask = morphClose(initialMask, width, height, 3);
	const cleanedMask = morphOpen(closedMask, width, height, 2);

	// Keep only the largest connected component (the lesion)
	const mask = largestConnectedComponent(cleanedMask, width, height);

	// Compute bounding box, area, and perimeter
	let minX = width;
	let minY = height;
	let maxX = 0;
	let maxY = 0;
	let area = 0;
	let perimeter = 0;

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const idx = y * width + x;
			if (mask[idx] !== 1) continue;

			area++;
			if (x < minX) minX = x;
			if (x > maxX) maxX = x;
			if (y < minY) minY = y;
			if (y > maxY) maxY = y;

			// Check if this is a border pixel (has at least one background neighbor)
			let isBorder = false;
			for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
				const nx = x + dx;
				const ny = y + dy;
				if (nx < 0 || nx >= width || ny < 0 || ny >= height || mask[ny * width + nx] === 0) {
					isBorder = true;
					break;
				}
			}
			if (isBorder) perimeter++;
		}
	}

	// Fallback: if segmentation fails (too small or empty), use center ellipse
	if (area < totalPixels * 0.01) {
		return fallbackSegmentation(width, height);
	}

	return {
		mask,
		bbox: {
			x: minX,
			y: minY,
			w: Math.max(1, maxX - minX + 1),
			h: Math.max(1, maxY - minY + 1),
		},
		area,
		perimeter,
	};
}

/**
 * Fallback segmentation using a centered ellipse when Otsu fails.
 * Assumes the lesion occupies roughly the center of the dermoscopic image.
 */
function fallbackSegmentation(width: number, height: number): SegmentationResult {
	const cx = width / 2;
	const cy = height / 2;
	const rx = width * 0.35;
	const ry = height * 0.35;

	const mask = new Uint8Array(width * height);
	let area = 0;
	let perimeter = 0;

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const dx = (x - cx) / rx;
			const dy = (y - cy) / ry;
			const dist = dx * dx + dy * dy;
			if (dist <= 1.0) {
				mask[y * width + x] = 1;
				area++;
				// Perimeter: near edge of ellipse
				if (dist > 0.85) perimeter++;
			}
		}
	}

	const bx = Math.max(0, Math.round(cx - rx));
	const by = Math.max(0, Math.round(cy - ry));

	return {
		mask,
		bbox: {
			x: bx,
			y: by,
			w: Math.min(width - bx, Math.round(rx * 2)),
			h: Math.min(height - by, Math.round(ry * 2)),
		},
		area,
		perimeter,
	};
}

// ============================================================
// 2. ASYMMETRY MEASUREMENT
// ============================================================

/**
 * Measure lesion asymmetry by folding along principal axes.
 *
 * Computes the centroid and principal axis of inertia for the
 * lesion mask, then folds the mask along both the principal axis
 * and its perpendicular. The asymmetry score is the percentage
 * of non-overlapping area.
 *
 * @returns Score 0 (symmetric) to 2 (highly asymmetric)
 */
export function measureAsymmetry(
	mask: Uint8Array,
	width: number,
	height: number,
	bbox: BBox,
): number {
	if (bbox.w === 0 || bbox.h === 0) return 0;

	// Compute centroid of the lesion
	let sumX = 0;
	let sumY = 0;
	let count = 0;
	for (let y = bbox.y; y < bbox.y + bbox.h; y++) {
		for (let x = bbox.x; x < bbox.x + bbox.w; x++) {
			if (mask[y * width + x] === 1) {
				sumX += x;
				sumY += y;
				count++;
			}
		}
	}

	if (count === 0) return 0;

	const cx = sumX / count;
	const cy = sumY / count;

	// Compute second-order central moments for principal axis
	let mxx = 0;
	let myy = 0;
	let mxy = 0;
	for (let y = bbox.y; y < bbox.y + bbox.h; y++) {
		for (let x = bbox.x; x < bbox.x + bbox.w; x++) {
			if (mask[y * width + x] === 1) {
				const dx = x - cx;
				const dy = y - cy;
				mxx += dx * dx;
				myy += dy * dy;
				mxy += dx * dy;
			}
		}
	}

	// Principal axis angle (angle of the eigenvector with largest eigenvalue)
	const theta = 0.5 * Math.atan2(2 * mxy, mxx - myy);

	// Measure asymmetry along both axes
	const asymAxis1 = measureAxisAsymmetry(mask, width, height, bbox, cx, cy, theta);
	const asymAxis2 = measureAxisAsymmetry(mask, width, height, bbox, cx, cy, theta + Math.PI / 2);

	// Score: 0 if both symmetric, 1 if one axis asymmetric, 2 if both
	const threshold = 0.15; // 15% non-overlap = asymmetric
	let score = 0;
	if (asymAxis1 > threshold) score++;
	if (asymAxis2 > threshold) score++;

	return score;
}

/**
 * Measure asymmetry along a single axis defined by angle theta through (cx, cy).
 * Returns the fraction of non-overlapping area when folded along that axis.
 */
function measureAxisAsymmetry(
	mask: Uint8Array,
	width: number,
	height: number,
	bbox: BBox,
	cx: number,
	cy: number,
	theta: number,
): number {
	const cosT = Math.cos(theta);
	const sinT = Math.sin(theta);

	let mismatch = 0;
	let total = 0;

	for (let y = bbox.y; y < bbox.y + bbox.h; y++) {
		for (let x = bbox.x; x < bbox.x + bbox.w; x++) {
			// Project onto axis perpendicular to theta
			const dx = x - cx;
			const dy = y - cy;
			const perpDist = -dx * sinT + dy * cosT;

			// Only consider one side (perpDist >= 0)
			if (perpDist < 0) continue;

			// Find mirror point across the axis
			const mirrorX = Math.round(x + 2 * perpDist * sinT);
			const mirrorY = Math.round(y - 2 * perpDist * cosT);

			if (mirrorX < 0 || mirrorX >= width || mirrorY < 0 || mirrorY >= height) {
				// Mirror falls outside image -- count as mismatch if this pixel is lesion
				if (mask[y * width + x] === 1) {
					total++;
					mismatch++;
				}
				continue;
			}

			const original = mask[y * width + x];
			const mirrored = mask[mirrorY * width + mirrorX];
			if (original === 1 || mirrored === 1) {
				total++;
				if (original !== mirrored) mismatch++;
			}
		}
	}

	return total > 0 ? mismatch / total : 0;
}

// ============================================================
// 3. BORDER ANALYSIS
// ============================================================

/**
 * Analyze border irregularity across 8 octants.
 *
 * Divides the lesion border into 8 angular segments, then measures
 * the variation of border radii within each segment. Segments with
 * high coefficient of variation (abrupt distance changes) are scored
 * as irregular. Also checks for abrupt pigment transitions at borders.
 *
 * @returns Score 0 (smooth, regular border) to 8 (all segments irregular)
 */
export function analyzeBorder(
	imageData: ImageData,
	mask: Uint8Array,
	width: number,
	height: number,
): number {
	const { data } = imageData;

	// Find border pixels and lesion centroid
	const borderPixels: Array<{ x: number; y: number; angle: number; radius: number }> = [];
	let cxSum = 0;
	let cySum = 0;
	let lesionCount = 0;

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			if (mask[y * width + x] === 1) {
				cxSum += x;
				cySum += y;
				lesionCount++;
			}
		}
	}

	if (lesionCount === 0) return 0;

	const cx = cxSum / lesionCount;
	const cy = cySum / lesionCount;

	// Collect border pixels
	for (let y = 1; y < height - 1; y++) {
		for (let x = 1; x < width - 1; x++) {
			if (mask[y * width + x] !== 1) continue;

			let isBorder = false;
			for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
				const nx = x + dx;
				const ny = y + dy;
				if (mask[ny * width + nx] === 0) {
					isBorder = true;
					break;
				}
			}

			if (isBorder) {
				const angle = Math.atan2(y - cy, x - cx);
				const radius = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
				borderPixels.push({ x, y, angle, radius });
			}
		}
	}

	if (borderPixels.length < 16) return 0;

	// Divide into 8 octants
	const octants: Array<{ radii: number[]; pixelIndices: number[] }> = Array.from(
		{ length: 8 },
		() => ({ radii: [], pixelIndices: [] }),
	);

	for (let i = 0; i < borderPixels.length; i++) {
		const bp = borderPixels[i];
		let normalizedAngle = bp.angle + Math.PI; // [0, 2*PI]
		const octIdx = Math.min(7, Math.floor((normalizedAngle / (2 * Math.PI)) * 8));
		octants[octIdx].radii.push(bp.radius);
		octants[octIdx].pixelIndices.push(i);
	}

	let irregularCount = 0;

	for (const oct of octants) {
		if (oct.radii.length < 3) continue;

		const radii = oct.radii;
		const mean = radii.reduce((a, b) => a + b, 0) / radii.length;
		if (mean < 1) continue;

		const variance = radii.reduce((a, b) => a + (b - mean) ** 2, 0) / radii.length;
		const cv = Math.sqrt(variance) / mean;

		// Also check color gradient at border pixels
		let gradientSum = 0;
		let gradientCount = 0;
		for (const pi of oct.pixelIndices) {
			const bp = borderPixels[pi];
			const px = bp.y * width + bp.x;
			// Compare lesion side to background side
			for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
				const nx = bp.x + dx;
				const ny = bp.y + dy;
				if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
				if (mask[ny * width + nx] === 0) {
					// This neighbor is background -- compute color difference
					const lesionPx = px * 4;
					const bgPx = (ny * width + nx) * 4;
					const dr = data[lesionPx] - data[bgPx];
					const dg = data[lesionPx + 1] - data[bgPx + 1];
					const db = data[lesionPx + 2] - data[bgPx + 2];
					gradientSum += Math.sqrt(dr * dr + dg * dg + db * db);
					gradientCount++;
				}
			}
		}

		const avgGradient = gradientCount > 0 ? gradientSum / gradientCount : 0;

		// Irregular if high shape variation OR abrupt color transitions
		if (cv > 0.25 || avgGradient > 80) {
			irregularCount++;
		}
	}

	return irregularCount;
}

// ============================================================
// 4. COLOR ANALYSIS
// ============================================================

/** Dermoscopic reference colors in LAB space */
const DERM_COLORS: Array<{ name: string; lab: [number, number, number]; rgb: [number, number, number] }> = [
	{ name: "white", lab: [95, 0, 0], rgb: [240, 240, 240] },
	{ name: "red", lab: [50, 55, 35], rgb: [190, 60, 50] },
	{ name: "light-brown", lab: [55, 15, 30], rgb: [170, 120, 70] },
	{ name: "dark-brown", lab: [30, 15, 20], rgb: [90, 50, 30] },
	{ name: "blue-gray", lab: [55, -5, -15], rgb: [120, 140, 165] },
	{ name: "black", lab: [10, 0, 0], rgb: [25, 25, 25] },
];

/**
 * Analyze the color composition of the lesion using k-means clustering
 * in LAB color space, then map clusters to standard dermoscopic colors.
 *
 * @returns Color count, dominant colors with percentages, and blue-white structure flag
 */
export function analyzeColors(imageData: ImageData, mask: Uint8Array): ColorAnalysisResult {
	const { data, width, height } = imageData;

	// Collect LAB values for lesion pixels (sample for performance)
	const labPixels: Array<[number, number, number]> = [];
	const rgbPixels: Array<[number, number, number]> = [];
	const step = Math.max(1, Math.floor(width * height / 5000)); // Sample up to ~5000 pixels

	for (let i = 0; i < width * height; i += step) {
		if (mask[i] !== 1) continue;
		const px = i * 4;
		const r = data[px];
		const g = data[px + 1];
		const b = data[px + 2];
		labPixels.push(rgbToLab(r, g, b));
		rgbPixels.push([r, g, b]);
	}

	if (labPixels.length === 0) {
		return {
			colorCount: 1,
			dominantColors: [{ name: "light-brown", percentage: 100, rgb: [170, 120, 70] }],
			hasBlueWhiteStructures: false,
		};
	}

	// K-means clustering in LAB space (k=6 to match dermoscopic colors)
	const k = Math.min(6, labPixels.length);
	const assignments = kMeansLab(labPixels, k, 15);

	// Compute cluster centroids and sizes
	const clusterSums: Array<[number, number, number]> = Array.from({ length: k }, () => [0, 0, 0]);
	const clusterRgbSums: Array<[number, number, number]> = Array.from({ length: k }, () => [0, 0, 0]);
	const clusterCounts = new Int32Array(k);

	for (let i = 0; i < labPixels.length; i++) {
		const c = assignments[i];
		clusterSums[c][0] += labPixels[i][0];
		clusterSums[c][1] += labPixels[i][1];
		clusterSums[c][2] += labPixels[i][2];
		clusterRgbSums[c][0] += rgbPixels[i][0];
		clusterRgbSums[c][1] += rgbPixels[i][1];
		clusterRgbSums[c][2] += rgbPixels[i][2];
		clusterCounts[c]++;
	}

	// Map each cluster to the nearest dermoscopic color
	const colorPresence = new Map<string, { count: number; rgb: [number, number, number] }>();
	const totalSampled = labPixels.length;

	for (let c = 0; c < k; c++) {
		if (clusterCounts[c] === 0) continue;

		const centroidL = clusterSums[c][0] / clusterCounts[c];
		const centroidA = clusterSums[c][1] / clusterCounts[c];
		const centroidB = clusterSums[c][2] / clusterCounts[c];
		const centroidR = Math.round(clusterRgbSums[c][0] / clusterCounts[c]);
		const centroidG = Math.round(clusterRgbSums[c][1] / clusterCounts[c]);
		const centroidBv = Math.round(clusterRgbSums[c][2] / clusterCounts[c]);

		// Find nearest dermoscopic color
		let bestDist = Infinity;
		let bestColor = DERM_COLORS[0];
		for (const dc of DERM_COLORS) {
			const dL = centroidL - dc.lab[0];
			const dA = centroidA - dc.lab[1];
			const dB = centroidB - dc.lab[2];
			const dist = dL * dL + dA * dA + dB * dB;
			if (dist < bestDist) {
				bestDist = dist;
				bestColor = dc;
			}
		}

		// Only assign if reasonably close (delta E < 40 in LAB)
		if (bestDist < 1600) {
			const existing = colorPresence.get(bestColor.name);
			if (existing) {
				existing.count += clusterCounts[c];
			} else {
				colorPresence.set(bestColor.name, {
					count: clusterCounts[c],
					rgb: [centroidR, centroidG, centroidBv] as [number, number, number],
				});
			}
		}
	}

	// Build result -- only colors with >3% of lesion pixels
	const minThreshold = totalSampled * 0.03;
	const dominantColors: Array<{ name: string; percentage: number; rgb: [number, number, number] }> = [];

	for (const [name, info] of colorPresence) {
		if (info.count >= minThreshold) {
			dominantColors.push({
				name,
				percentage: Math.round((info.count / totalSampled) * 1000) / 10,
				rgb: info.rgb,
			});
		}
	}

	dominantColors.sort((a, b) => b.percentage - a.percentage);

	// Detect blue-white structures
	const hasBlueGray = colorPresence.has("blue-gray") &&
		(colorPresence.get("blue-gray")!.count / totalSampled) > 0.05;
	const hasWhite = colorPresence.has("white") &&
		(colorPresence.get("white")!.count / totalSampled) > 0.03;
	const hasBlueWhiteStructures = hasBlueGray && hasWhite;

	return {
		colorCount: Math.max(1, dominantColors.length),
		dominantColors: dominantColors.length > 0
			? dominantColors
			: [{ name: "light-brown", percentage: 100, rgb: [170, 120, 70] }],
		hasBlueWhiteStructures,
	};
}

/**
 * Simple k-means clustering in LAB space.
 * Returns cluster assignment for each pixel.
 */
function kMeansLab(
	pixels: Array<[number, number, number]>,
	k: number,
	maxIter: number,
): Int32Array {
	const n = pixels.length;
	const assignments = new Int32Array(n);

	// Initialize centroids using k-means++ seeding
	const centroids: Array<[number, number, number]> = [];
	centroids.push([...pixels[Math.floor(Math.random() * n)]]);

	for (let c = 1; c < k; c++) {
		// Compute distances to nearest centroid
		const dists = new Float64Array(n);
		let totalDist = 0;
		for (let i = 0; i < n; i++) {
			let minDist = Infinity;
			for (const cent of centroids) {
				const d = (pixels[i][0] - cent[0]) ** 2 +
					(pixels[i][1] - cent[1]) ** 2 +
					(pixels[i][2] - cent[2]) ** 2;
				if (d < minDist) minDist = d;
			}
			dists[i] = minDist;
			totalDist += minDist;
		}

		// Weighted random selection
		let target = Math.random() * totalDist;
		let chosen = 0;
		for (let i = 0; i < n; i++) {
			target -= dists[i];
			if (target <= 0) {
				chosen = i;
				break;
			}
		}
		centroids.push([...pixels[chosen]]);
	}

	// Iterate
	for (let iter = 0; iter < maxIter; iter++) {
		// Assign each pixel to nearest centroid
		let changed = false;
		for (let i = 0; i < n; i++) {
			let bestC = 0;
			let bestDist = Infinity;
			for (let c = 0; c < k; c++) {
				const d = (pixels[i][0] - centroids[c][0]) ** 2 +
					(pixels[i][1] - centroids[c][1]) ** 2 +
					(pixels[i][2] - centroids[c][2]) ** 2;
				if (d < bestDist) {
					bestDist = d;
					bestC = c;
				}
			}
			if (assignments[i] !== bestC) {
				assignments[i] = bestC;
				changed = true;
			}
		}

		if (!changed) break;

		// Update centroids
		const sums: Array<[number, number, number]> = Array.from({ length: k }, () => [0, 0, 0]);
		const counts = new Int32Array(k);
		for (let i = 0; i < n; i++) {
			const c = assignments[i];
			sums[c][0] += pixels[i][0];
			sums[c][1] += pixels[i][1];
			sums[c][2] += pixels[i][2];
			counts[c]++;
		}
		for (let c = 0; c < k; c++) {
			if (counts[c] > 0) {
				centroids[c][0] = sums[c][0] / counts[c];
				centroids[c][1] = sums[c][1] / counts[c];
				centroids[c][2] = sums[c][2] / counts[c];
			}
		}
	}

	return assignments;
}

// ============================================================
// 5. TEXTURE ANALYSIS (GLCM)
// ============================================================

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

// ============================================================
// 6. STRUCTURAL PATTERN DETECTION
// ============================================================

/**
 * Detect dermoscopic structures using Local Binary Pattern (LBP) analysis
 * and frequency-domain features.
 *
 * Structures detected:
 * - Pigment network (regular = benign, irregular = suspicious)
 * - Globules/dots (clustered round structures)
 * - Streaks/pseudopods (radial linear structures)
 * - Blue-white veil (diffuse blue-gray area)
 * - Regression structures (white scar-like areas)
 *
 * @returns Structure presence flags and overall structural suspicion score
 */
export function detectStructures(
	imageData: ImageData,
	mask: Uint8Array,
): StructureResult {
	const { data, width, height } = imageData;
	const gray = toGrayscale(imageData);

	// --- Compute LBP (Local Binary Pattern) ---
	const lbp = new Uint8Array(width * height);
	const lbpHistogram = new Float64Array(256);
	let lbpCount = 0;

	for (let y = 1; y < height - 1; y++) {
		for (let x = 1; x < width - 1; x++) {
			if (mask[y * width + x] !== 1) continue;

			const center = gray[y * width + x];
			let pattern = 0;

			// 8 neighbors clockwise starting from top-left
			const neighbors = [
				gray[(y - 1) * width + (x - 1)],
				gray[(y - 1) * width + x],
				gray[(y - 1) * width + (x + 1)],
				gray[y * width + (x + 1)],
				gray[(y + 1) * width + (x + 1)],
				gray[(y + 1) * width + x],
				gray[(y + 1) * width + (x - 1)],
				gray[y * width + (x - 1)],
			];

			for (let n = 0; n < 8; n++) {
				if (neighbors[n] >= center) {
					pattern |= (1 << n);
				}
			}

			lbp[y * width + x] = pattern;
			lbpHistogram[pattern]++;
			lbpCount++;
		}
	}

	// Normalize LBP histogram
	if (lbpCount > 0) {
		for (let i = 0; i < 256; i++) {
			lbpHistogram[i] /= lbpCount;
		}
	}

	// --- Detect pigment network (grid-like repetitive pattern) ---
	// Uniform LBP patterns indicate regular network; non-uniform indicate irregular
	let uniformCount = 0;
	let nonUniformCount = 0;
	for (let i = 0; i < 256; i++) {
		if (lbpHistogram[i] < 0.001) continue;
		// Count transitions (0->1 or 1->0) in the binary pattern
		let transitions = 0;
		for (let bit = 0; bit < 8; bit++) {
			const curr = (i >> bit) & 1;
			const next = (i >> ((bit + 1) % 8)) & 1;
			if (curr !== next) transitions++;
		}
		if (transitions <= 2) {
			uniformCount += lbpHistogram[i];
		} else {
			nonUniformCount += lbpHistogram[i];
		}
	}

	const hasIrregularNetwork = nonUniformCount > 0.35;

	// --- Detect globules/dots ---
	// Look for small round dark regions within the lesion using local min detection
	let globuleCount = 0;
	let irregularGlobules = 0;
	const radius = 4;

	for (let y = radius; y < height - radius; y += radius) {
		for (let x = radius; x < width - radius; x += radius) {
			if (mask[y * width + x] !== 1) continue;

			const centerGray = gray[y * width + x];
			let isLocalMin = true;
			let neighborSum = 0;
			let neighborCount = 0;
			let minNeighbor = 255;
			let maxNeighbor = 0;

			for (let dy = -radius; dy <= radius; dy++) {
				for (let dx = -radius; dx <= radius; dx++) {
					if (dx === 0 && dy === 0) continue;
					const nx = x + dx;
					const ny = y + dy;
					if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
					if (mask[ny * width + nx] !== 1) continue;

					const nGray = gray[ny * width + nx];
					neighborSum += nGray;
					neighborCount++;
					if (nGray < minNeighbor) minNeighbor = nGray;
					if (nGray > maxNeighbor) maxNeighbor = nGray;

					if (nGray <= centerGray) {
						isLocalMin = false;
					}
				}
			}

			// Dark spot that is significantly darker than surroundings
			if (neighborCount > 0 && !isLocalMin) continue;
			const avgNeighbor = neighborCount > 0 ? neighborSum / neighborCount : centerGray;
			if (centerGray < avgNeighbor - 20) {
				globuleCount++;
				// Irregular if the surrounding variance is high
				if (maxNeighbor - minNeighbor > 60) {
					irregularGlobules++;
				}
			}
		}
	}

	const hasIrregularGlobules = irregularGlobules > 3 &&
		(globuleCount > 0 ? irregularGlobules / globuleCount > 0.4 : false);

	// --- Detect streaks/pseudopods ---
	// Look for radial linear structures at the periphery using directional gradients
	let streakScore = 0;
	let streakSamples = 0;

	// Analyze border region (outer 20% of bounding box)
	let minX = width, minY = height, maxX = 0, maxY = 0;
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			if (mask[y * width + x] === 1) {
				if (x < minX) minX = x;
				if (x > maxX) maxX = x;
				if (y < minY) minY = y;
				if (y > maxY) maxY = y;
			}
		}
	}

	const bboxW = maxX - minX + 1;
	const bboxH = maxY - minY + 1;
	const borderThickness = Math.max(3, Math.round(Math.min(bboxW, bboxH) * 0.2));

	for (let y = minY; y <= maxY; y += 2) {
		for (let x = minX; x <= maxX; x += 2) {
			if (mask[y * width + x] !== 1) continue;

			// Check if near border
			const distToEdgeX = Math.min(x - minX, maxX - x);
			const distToEdgeY = Math.min(y - minY, maxY - y);
			if (Math.min(distToEdgeX, distToEdgeY) > borderThickness) continue;

			// Compute Sobel gradients
			if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) continue;
			const gx =
				-gray[(y - 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)] +
				-2 * gray[y * width + (x - 1)] + 2 * gray[y * width + (x + 1)] +
				-gray[(y + 1) * width + (x - 1)] + gray[(y + 1) * width + (x + 1)];
			const gy =
				-gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)] +
				gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];

			const magnitude = Math.sqrt(gx * gx + gy * gy);

			// Check if gradient direction is radial (pointing outward from center)
			const cx = (minX + maxX) / 2;
			const cy = (minY + maxY) / 2;
			const radialDx = x - cx;
			const radialDy = y - cy;
			const radialMag = Math.sqrt(radialDx * radialDx + radialDy * radialDy);

			if (radialMag > 0 && magnitude > 30) {
				const gradAngle = Math.atan2(gy, gx);
				const radialAngle = Math.atan2(radialDy, radialDx);
				// Perpendicular to radial = along the streak direction
				const angleDiff = Math.abs(Math.sin(gradAngle - radialAngle));
				if (angleDiff < 0.4) { // gradient roughly perpendicular to radial direction
					streakScore += magnitude;
				}
				streakSamples++;
			}
		}
	}

	const avgStreakScore = streakSamples > 0 ? streakScore / streakSamples : 0;
	const hasStreaks = avgStreakScore > 40;

	// --- Detect blue-white veil ---
	// Diffuse blue-white area within the lesion
	let blueWhiteCount = 0;
	let lesionPixelCount = 0;

	for (let i = 0; i < width * height; i++) {
		if (mask[i] !== 1) continue;
		lesionPixelCount++;

		const px = i * 4;
		const r = data[px];
		const g = data[px + 1];
		const b = data[px + 2];

		// Blue-white veil: blue-shifted with moderate brightness
		const brightness = (r + g + b) / 3;
		if (b > r && b > g && brightness > 80 && brightness < 200 && (b - r) > 20) {
			blueWhiteCount++;
		}
	}

	const hasBlueWhiteVeil = lesionPixelCount > 0 && (blueWhiteCount / lesionPixelCount) > 0.1;

	// --- Detect regression structures ---
	// White scar-like areas (high brightness, low saturation within lesion)
	let regressionCount = 0;

	for (let i = 0; i < width * height; i++) {
		if (mask[i] !== 1) continue;
		const px = i * 4;
		const r = data[px];
		const g = data[px + 1];
		const b = data[px + 2];

		const maxC = Math.max(r, g, b);
		const minC = Math.min(r, g, b);
		const saturation = maxC > 0 ? (maxC - minC) / maxC : 0;
		const brightness = (r + g + b) / 3;

		// White/pink desaturated areas within the lesion
		if (brightness > 180 && saturation < 0.15) {
			regressionCount++;
		}
	}

	const hasRegressionStructures = lesionPixelCount > 0 &&
		(regressionCount / lesionPixelCount) > 0.08;

	// --- Overall structural score ---
	let structuralScore = 0;
	const weights = {
		irregularNetwork: 0.2,
		irregularGlobules: 0.15,
		streaks: 0.25,
		blueWhiteVeil: 0.25,
		regression: 0.15,
	};

	if (hasIrregularNetwork) structuralScore += weights.irregularNetwork;
	if (hasIrregularGlobules) structuralScore += weights.irregularGlobules;
	if (hasStreaks) structuralScore += weights.streaks;
	if (hasBlueWhiteVeil) structuralScore += weights.blueWhiteVeil;
	if (hasRegressionStructures) structuralScore += weights.regression;

	return {
		hasIrregularNetwork,
		hasIrregularGlobules,
		hasStreaks,
		hasBlueWhiteVeil,
		hasRegressionStructures,
		structuralScore,
	};
}

// ============================================================
// 7. ATTENTION HEATMAP
// ============================================================

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
function gaussianSmooth(data: Float32Array, w: number, h: number, sigma: number): Float32Array {
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
function resizeFloat32(
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

// ============================================================
// 8. COMBINED FEATURE CLASSIFICATION
// ============================================================

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
	// TDS (Total Dermoscopy Score) — weighted ABCD formula
	// Calibrated against DermaSensor DEN230008 + dermoscopy literature
	// TDS = A×1.3 + B×0.1 + C×0.5 + D×0.5
	// < 4.75 = benign, 4.75-5.45 = suspicious, > 5.45 = malignant
	// ============================================================
	const tds = asymmetry * 1.3 + borderScore * 0.1 + colorCount * 0.5 + structuralScore * 3.0 * 0.5;
	const tdsSuspicious = tds >= 4.75;
	const tdsMalignant = tds > 5.45;

	// ============================================================
	// Melanoma safety gate (calibrated to 95% sensitivity target)
	// DermaSensor achieves 90.2% melanoma sens, we target 95%
	// A melanoma requires AT LEAST 2 concurrent suspicious features
	// If 3+ features are suspicious → melanoma floor of 15%
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

// ============================================================
// Utility: Estimate physical diameter from pixel area
// ============================================================

/**
 * Estimate lesion diameter in millimeters from pixel area.
 *
 * Assumes a standard dermoscope field of view of ~25mm at 10x
 * and that the image represents the full field.
 *
 * @param areaPixels - Lesion area in pixels
 * @param imageWidth - Image width in pixels
 * @param magnification - Dermoscope magnification (default 10x)
 * @returns Estimated diameter in millimeters
 */
export function estimateDiameterMm(
	areaPixels: number,
	imageWidth: number,
	magnification: number = 10,
): number {
	// Standard dermoscope: ~25mm field of view at 10x
	const fieldOfViewMm = 25 / (magnification / 10);
	const pxPerMm = imageWidth / fieldOfViewMm;
	const radiusPx = Math.sqrt(areaPixels / Math.PI);
	const diameterMm = (2 * radiusPx) / pxPerMm;
	return Math.round(diameterMm * 10) / 10;
}

// ============================================================
// Utility: Risk level from ABCDE scores
// ============================================================

/**
 * Compute risk level from real ABCDE feature values.
 *
 * Based on the ABCDE melanoma screening rule:
 * - Asymmetry > 1 = +1 risk point
 * - Border > 4 irregular segments = +1 risk point
 * - Color count >= 3 = +1 risk point
 * - Diameter > 6mm = +1 risk point
 *
 * @returns Risk level: low, moderate, high, or critical
 */
export function computeRiskLevel(
	asymmetry: number,
	borderScore: number,
	colorCount: number,
	diameterMm: number = 0,
): "low" | "moderate" | "high" | "critical" {
	let riskPoints = 0;

	if (asymmetry >= 1) riskPoints++;
	if (asymmetry >= 2) riskPoints++;
	if (borderScore >= 4) riskPoints++;
	if (borderScore >= 6) riskPoints++;
	if (colorCount >= 3) riskPoints++;
	if (colorCount >= 5) riskPoints++;
	if (diameterMm > 6) riskPoints++;

	if (riskPoints <= 1) return "low";
	if (riskPoints <= 3) return "moderate";
	if (riskPoints <= 5) return "high";
	return "critical";
}

// ============================================================
// LESION PRESENCE DETECTION (Safety Gate)
// ============================================================

export interface LesionPresenceResult {
	hasLesion: boolean;
	confidence: number;
	reason: string;
}

/**
 * Detect whether the image actually contains a skin lesion.
 * Returns a confidence score (0-1) that a lesion is present.
 *
 * This is a SAFETY GATE: normal skin (e.g. a hand with no moles)
 * must NOT proceed to classification, which would spuriously
 * label it as melanoma because the model was trained only on
 * lesion images.
 *
 * Checks:
 * 1. Segmentation quality: did Otsu find a meaningful boundary?
 * 2. Color contrast: is there a distinct lesion vs surrounding skin?
 * 3. Size: is the segmented area reasonable (not too small, not the whole image)?
 * 4. Shape: does it look like a lesion (roughly circular/oval)?
 */
export function detectLesionPresence(imageData: ImageData): LesionPresenceResult {
	const { width, height } = imageData;
	const totalPixels = width * height;

	// Run segmentation
	const seg = segmentLesion(imageData);

	// Check 1: Area ratio — lesion should be 2-60% of image
	const areaRatio = seg.area / totalPixels;
	if (areaRatio < 0.02) {
		return { hasLesion: false, confidence: 0.2, reason: "No distinct lesion detected — area too small" };
	}
	if (areaRatio > 0.85) {
		return { hasLesion: false, confidence: 0.3, reason: "No distinct lesion boundary — image may not contain a focused lesion" };
	}

	// Check 2: Compactness — real lesions are roughly circular
	const expectedPerimeter = 2 * Math.sqrt(Math.PI * seg.area);
	const compactness = expectedPerimeter / Math.max(seg.perimeter, 1);
	// Very low compactness = irregular/fragmented = probably not a real lesion
	if (compactness < 0.2) {
		return { hasLesion: false, confidence: 0.3, reason: "Segmentation is fragmented — no clear lesion boundary" };
	}

	// Check 3: Color contrast between lesion and surrounding skin
	const data = imageData.data;
	let lesionR = 0, lesionG = 0, lesionB = 0, lesionCount = 0;
	let skinR = 0, skinG = 0, skinB = 0, skinCount = 0;

	for (let y = 0; y < height; y += 2) {
		for (let x = 0; x < width; x += 2) {
			const i = (y * width + x) * 4;
			const maskIdx = y * width + x;
			if (seg.mask[maskIdx]) {
				lesionR += data[i]; lesionG += data[i + 1]; lesionB += data[i + 2];
				lesionCount++;
			} else {
				skinR += data[i]; skinG += data[i + 1]; skinB += data[i + 2];
				skinCount++;
			}
		}
	}

	let colorDiff = 0;
	if (lesionCount > 0 && skinCount > 0) {
		lesionR /= lesionCount; lesionG /= lesionCount; lesionB /= lesionCount;
		skinR /= skinCount; skinG /= skinCount; skinB /= skinCount;

		colorDiff = Math.sqrt(
			(lesionR - skinR) ** 2 + (lesionG - skinG) ** 2 + (lesionB - skinB) ** 2
		);

		// Low color difference = no distinct lesion
		if (colorDiff < 15) {
			return { hasLesion: false, confidence: 0.25, reason: "No color contrast between lesion and skin — may be normal skin" };
		}
	}

	// All checks passed — compute overall confidence
	const lesionConfidence = Math.min(1,
		(areaRatio > 0.05 ? 0.3 : 0.1) +
		(compactness > 0.4 ? 0.3 : 0.1) +
		(colorDiff > 30 ? 0.4 : 0.2)
	);

	return {
		hasLesion: lesionConfidence > 0.5,
		confidence: lesionConfidence,
		reason: lesionConfidence > 0.5 ? "Lesion detected" : "Uncertain — image may not contain a clear lesion",
	};
}
