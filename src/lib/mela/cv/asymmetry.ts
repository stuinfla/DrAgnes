/**
 * Lesion asymmetry measurement via principal-axis folding.
 */

import type { BBox } from "./types";

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
