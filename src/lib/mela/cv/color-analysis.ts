/**
 * Color analysis using k-means clustering in LAB space,
 * mapped to standard dermoscopic reference colors.
 */

import type { ColorAnalysisResult } from "./types";
import { rgbToLab } from "./color-space";

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
export function kMeansLab(
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
