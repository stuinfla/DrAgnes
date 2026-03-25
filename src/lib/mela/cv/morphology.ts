/**
 * Morphological image operations and connected-component analysis.
 *
 * Single source of truth for Otsu thresholding, dilate/erode,
 * open/close, and largest-connected-component extraction.
 */

export function otsuThreshold(values: Uint8Array | Float32Array, count: number): number {
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

export function largestConnectedComponent(mask: Uint8Array, width: number, height: number): Uint8Array {
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

export function morphDilate(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
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

export function morphErode(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
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

export function morphClose(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
	return morphErode(morphDilate(mask, w, h, r), w, h, r);
}

export function morphOpen(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
	return morphDilate(morphErode(mask, w, h, r), w, h, r);
}
