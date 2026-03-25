/**
 * Border irregularity analysis across 8 octants.
 *
 * Divides the lesion border into 8 angular segments, then measures
 * the variation of border radii within each segment. Segments with
 * high coefficient of variation (abrupt distance changes) are scored
 * as irregular. Also checks for abrupt pigment transitions at borders.
 */

/**
 * Analyze border irregularity across 8 octants.
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
