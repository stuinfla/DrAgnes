/**
 * Dermoscopic structural pattern detection using LBP and frequency analysis.
 *
 * Detects pigment network, globules/dots, streaks/pseudopods,
 * blue-white veil, and regression structures.
 */

import type { StructureResult } from "./types";
import { toGrayscale } from "./color-space";

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
