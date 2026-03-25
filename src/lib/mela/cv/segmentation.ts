/**
 * Lesion segmentation using LAB color space + Otsu + morphology.
 *
 * Segments the lesion from surrounding skin using L-channel thresholding,
 * morphological cleanup, and largest connected component extraction.
 */

import type { SegmentationResult } from "./types";
import { rgbToLab } from "./color-space";
import { otsuThreshold, morphClose, morphOpen, largestConnectedComponent } from "./morphology";

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
