/**
 * Two-pass hybrid spot detection. Browser-safe, zero native deps.
 * Pass 1: LAB histogram dark-tail test (< 5 ms)
 * Pass 2: Morphological blob detection + compactness/contrast validation (< 20 ms)
 */
export interface SpotDetection {
	hasSpot: boolean;
	confidence: number;
	spotCount: number;
	largestSpotArea: number;
	reason: string;
}

const TAIL_OFFSET = 40, TAIL_MIN = 0.0008, TAIL_MAX = 0.30;
const SPOT_MIN = 0.0004, SPOT_MAX = 0.20, COMPACT_MIN = 0.30, CONTRAST_MIN = 20;

function rgbToL(r: number, g: number, b: number): number {
	let rn = r / 255, gn = g / 255, bn = b / 255;
	rn = rn > 0.04045 ? ((rn + 0.055) / 1.055) ** 2.4 : rn / 12.92;
	gn = gn > 0.04045 ? ((gn + 0.055) / 1.055) ** 2.4 : gn / 12.92;
	bn = bn > 0.04045 ? ((bn + 0.055) / 1.055) ** 2.4 : bn / 12.92;
	let y = rn * 0.2126729 + gn * 0.7151522 + bn * 0.0721750;
	y = y > 0.008856 ? Math.cbrt(y) : (903.3 * y + 16) / 116;
	return 116 * y - 16;
}

function dilate(m: Uint8Array, w: number, h: number): Uint8Array {
	const o = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
		let v = 0;
		for (let dy = -1; dy <= 1 && !v; dy++) for (let dx = -1; dx <= 1 && !v; dx++) {
			const ny = y + dy, nx = x + dx;
			if (ny >= 0 && ny < h && nx >= 0 && nx < w && m[ny * w + nx]) v = 1;
		}
		o[y * w + x] = v;
	}
	return o;
}

function erode(m: Uint8Array, w: number, h: number): Uint8Array {
	const o = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
		let v = 1;
		for (let dy = -1; dy <= 1 && v; dy++) for (let dx = -1; dx <= 1 && v; dx++) {
			const ny = y + dy, nx = x + dx;
			if (ny < 0 || ny >= h || nx < 0 || nx >= w || !m[ny * w + nx]) v = 0;
		}
		o[y * w + x] = v;
	}
	return o;
}

interface Comp { area: number; perim: number; sumL: number; px: number[] }
const N4: [number, number][] = [[0,1],[0,-1],[1,0],[-1,0]];
const N8: [number, number][] = [...N4,[1,1],[1,-1],[-1,1],[-1,-1]];

function findComponents(mask: Uint8Array, lMap: Float64Array, w: number, h: number): Comp[] {
	const vis = new Uint8Array(w * h), out: Comp[] = [];
	for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
		const idx = y * w + x;
		if (!mask[idx] || vis[idx]) continue;
		const px: number[] = [], q: number[] = [idx];
		vis[idx] = 1;
		let perim = 0, sumL = 0;
		while (q.length) {
			const c = q.pop()!; px.push(c); sumL += lMap[c];
			const cy = (c / w) | 0, cx = c % w;
			let border = false;
			for (const [dx, dy] of N4) {
				const nx = cx + dx, ny = cy + dy;
				if (nx < 0 || nx >= w || ny < 0 || ny >= h) { border = true; continue; }
				const ni = ny * w + nx;
				if (!mask[ni]) { border = true; continue; }
				if (!vis[ni]) { vis[ni] = 1; q.push(ni); }
			}
			if (border) perim++;
		}
		out.push({ area: px.length, perim, sumL, px });
	}
	return out;
}

function ringContrast(c: Comp, lMap: Float64Array, w: number, h: number): number {
	const inner = c.sumL / c.area, inSet = new Set(c.px);
	let rS = 0, rN = 0;
	for (const p of c.px) {
		const cy = (p / w) | 0, cx = p % w;
		for (const [dx, dy] of N8) {
			const nx = cx + dx, ny = cy + dy;
			if (nx >= 0 && nx < w && ny >= 0 && ny < h && !inSet.has(ny * w + nx)) { rS += lMap[ny * w + nx]; rN++; }
		}
	}
	return rN ? (rS / rN) - inner : 0;
}

export function detectSpots(imageData: ImageData): SpotDetection {
	const { data, width: w, height: h } = imageData;
	const total = w * h;

	// Pass 1: histogram dark-tail
	const hist = new Float64Array(101);
	for (let i = 0; i < total; i++) {
		const p = i * 4;
		hist[Math.max(0, Math.min(100, Math.round(rgbToL(data[p], data[p + 1], data[p + 2]))))]++;
	}
	let mode = 0, mc = 0;
	for (let i = 0; i < 101; i++) if (hist[i] > mc) { mode = i; mc = hist[i]; }
	const thresh = Math.max(0, mode - TAIL_OFFSET);
	let tailPx = 0;
	for (let i = 0; i < thresh; i++) tailPx += hist[i];
	const tailMass = tailPx / total;

	if (tailMass <= TAIL_MIN) return {
		hasSpot: false, confidence: 0.92, spotCount: 0, largestSpotArea: 0,
		reason: "Your skin looks healthy here. No spots or moles detected.",
	};

	// Pass 2: build dark mask -> morph close -> find & validate blobs
	const lMap = new Float64Array(total), mask = new Uint8Array(total);
	for (let i = 0; i < total; i++) {
		const p = i * 4, L = rgbToL(data[p], data[p + 1], data[p + 2]);
		lMap[i] = L; mask[i] = L < thresh ? 1 : 0;
	}
	const cleaned = erode(dilate(mask, w, h), w, h); // close only; open kills tiny spots
	const comps = findComponents(cleaned, lMap, w, h);

	const valid: { area: number; compact: number; contrast: number }[] = [];
	for (const c of comps) {
		const frac = c.area / total;
		if (frac < SPOT_MIN || frac > SPOT_MAX) continue;
		const compact = c.perim > 0 ? (4 * Math.PI * c.area) / (c.perim ** 2) : 0;
		if (compact < COMPACT_MIN) continue;
		const contrast = ringContrast(c, lMap, w, h);
		if (contrast < CONTRAST_MIN) continue;
		valid.push({ area: c.area, compact, contrast });
	}

	if (!valid.length) return {
		hasSpot: false, confidence: 0.70, spotCount: 0, largestSpotArea: 0,
		reason: "No distinct spots found. Dark regions appear to be shadows or skin texture.",
	};

	valid.sort((a, b) => b.area - a.area);
	const best = valid[0];
	const conf = Math.min(1,
		Math.min(0.35, tailMass * 20) +
		(best.compact > 0.5 ? 0.35 : best.compact > 0.35 ? 0.25 : 0.15) +
		Math.min(0.30, (best.contrast - CONTRAST_MIN) / 80 * 0.30));

	return {
		hasSpot: true, confidence: conf, spotCount: valid.length, largestSpotArea: best.area,
		reason: valid.length === 1 ? "Spot detected. Proceeding to analysis."
			: `${valid.length} spots detected. Proceeding to analysis.`,
	};
}
