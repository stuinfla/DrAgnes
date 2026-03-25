/**
 * USB-C / Lightning / USB-A Connector Reference Detection -- ADR-121 Phase 2
 *
 * Detects a connector in a skin photo and uses its known dimensions to
 * calibrate pixels-per-mm for real-world lesion size measurement.
 * Pure TypeScript, no native deps, runs in browser <200ms.
 */

export interface ConnectorBoundingBox {
	x: number; y: number; width: number; height: number;
}

export interface ConnectorDetection {
	detected: boolean;
	type: "usb-c" | "lightning" | "usb-a" | null;
	confidence: number; // 0-1
	pixelsPerMm: number;
	boundingBox: ConnectorBoundingBox | null;
	diameterMm: number | null;
}

interface ConnectorSpec {
	type: "usb-c" | "lightning" | "usb-a";
	widthMm: number;
	aspect: number;
}

const CONNECTORS: ConnectorSpec[] = [
	{ type: "usb-c", widthMm: 8.25, aspect: 8.25 / 2.4 },   // 3.44:1
	{ type: "lightning", widthMm: 7.7, aspect: 7.7 / 1.5 },  // 5.13:1
	{ type: "usb-a", widthMm: 12.0, aspect: 12.0 / 4.5 },    // 2.67:1
];

const ASPECT_TOL = 0.20;
const MIN_FRAC = 0.02;
const MAX_FRAC = 0.15;
const EDGE_T = 60;
const S = 2; // sample every 2nd pixel

interface Rect { x: number; y: number; w: number; h: number }

/** RGBA to grayscale (integer luminance). */
function toGray(d: Uint8Array | Uint8ClampedArray, w: number, h: number): Uint8Array {
	const g = new Uint8Array(w * h);
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++) {
			const i = (y * w + x) * 4;
			g[y * w + x] = (d[i] * 77 + d[i + 1] * 150 + d[i + 2] * 29) >> 8;
		}
	return g;
}

/** Sobel edge magnitude, sampled every S pixels. */
function sobel(g: Uint8Array, w: number, h: number): Uint8Array {
	const e = new Uint8Array(w * h);
	for (let y = 1; y < h - 1; y += S)
		for (let x = 1; x < w - 1; x += S) {
			const tl = g[(y-1)*w+x-1], tc = g[(y-1)*w+x], tr = g[(y-1)*w+x+1];
			const ml = g[y*w+x-1], mr = g[y*w+x+1];
			const bl = g[(y+1)*w+x-1], bc = g[(y+1)*w+x], br = g[(y+1)*w+x+1];
			const gx = -tl + tr - 2*ml + 2*mr - bl + br;
			const gy = -tl - 2*tc - tr + bl + 2*bc + br;
			e[y * w + x] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
		}
	return e;
}

/** Scan downward from a horizontal edge run to find the closing edge. */
function scanDown(e: Uint8Array, w: number, h: number, sx: number, sy: number, len: number): number {
	const maxH = Math.min(h - sy, w * MAX_FRAC);
	const minMatch = len * 0.5;
	for (let dy = 6; dy < maxH; dy += S) {
		if (sy + dy >= h) break;
		let cnt = 0;
		for (let x = sx; x < sx + len; x += S)
			if (e[(sy + dy) * w + x] > EDGE_T) cnt++;
		if (cnt * S >= minMatch) return dy;
	}
	return 0;
}

/** Find rectangular edge-bounded regions. */
function findRects(e: Uint8Array, w: number, h: number): Rect[] {
	const minRun = Math.max(8, w * MIN_FRAC * 0.5);
	const vis = new Uint8Array(w * h);
	const rects: Rect[] = [];
	for (let y = 2; y < h - 2; y += S) {
		let rs = -1;
		for (let x = 0; x < w; x++) {
			const isE = e[y * w + x] > EDGE_T;
			if (isE && rs === -1) { rs = x; }
			else if (!isE && rs !== -1) {
				const len = x - rs;
				if (len >= minRun && !vis[y * w + rs]) {
					const rh = scanDown(e, w, h, rs, y, len);
					if (rh > 4) {
						rects.push({ x: rs, y, w: len, h: rh });
						for (let dy = 0; dy < rh; dy += S) vis[(y + dy) * w + rs] = 1;
					}
				}
				rs = -1;
			}
		}
	}
	return rects;
}

/** Check if region has metallic color (low saturation, medium brightness). */
function isMetallic(d: Uint8Array | Uint8ClampedArray, imgW: number, r: Rect): boolean {
	let satS = 0, briS = 0, n = 0;
	for (let y = r.y + 2; y < r.y + r.h - 2; y += S * 2)
		for (let x = r.x + 2; x < r.x + r.w - 2; x += S * 2) {
			const i = (y * imgW + x) * 4;
			const mx = Math.max(d[i], d[i+1], d[i+2]);
			const mn = Math.min(d[i], d[i+1], d[i+2]);
			satS += mx === 0 ? 0 : (mx - mn) / mx;
			briS += mx / 255;
			n++;
		}
	if (n === 0) return false;
	return satS / n < 0.35 && briS / n > 0.15 && briS / n < 0.85;
}

/** Score proximity to image edge (0-1, higher = closer). */
function edgeProx(r: Rect, w: number, h: number): number {
	const dx = Math.min(r.x + r.w / 2, w - r.x - r.w / 2) / w;
	const dy = Math.min(r.y + r.h / 2, h - r.y - r.h / 2) / h;
	return 1 - Math.min(1, Math.min(dx, dy) * 4);
}

const NO_MATCH: ConnectorDetection = {
	detected: false, type: null, confidence: 0,
	pixelsPerMm: 0, boundingBox: null, diameterMm: null,
};

export function detectConnector(imageData: ImageData): ConnectorDetection {
	const { data, width: W, height: H } = imageData;
	const gray = toGray(data, W, H);
	const edges = sobel(gray, W, H);
	const rects = findRects(edges, W, H);

	let best: ConnectorDetection = { ...NO_MATCH };
	let bestScore = 0;

	for (const rect of rects) {
		const aspect = rect.w / rect.h;
		const frac = rect.w / W;
		if (frac < MIN_FRAC || frac > MAX_FRAC) continue;

		for (const spec of CONNECTORS) {
			const err = Math.abs(aspect - spec.aspect) / spec.aspect;
			if (err > ASPECT_TOL) continue;
			if (!isMetallic(data, W, rect)) continue;

			const aScore = 1 - err / ASPECT_TOL;
			const pScore = edgeProx(rect, W, H);
			const mid = (MIN_FRAC + MAX_FRAC) / 2;
			const sScore = 1 - Math.abs(frac - mid) / mid;
			const conf = Math.min(0.98, aScore * 0.50 + pScore * 0.25 + sScore * 0.25);

			if (conf > bestScore && conf > 0.3) {
				bestScore = conf;
				best = {
					detected: true,
					type: spec.type,
					confidence: Math.round(conf * 100) / 100,
					pixelsPerMm: Math.round((rect.w / spec.widthMm) * 100) / 100,
					boundingBox: { x: rect.x, y: rect.y, width: rect.w, height: rect.h },
					diameterMm: null,
				};
			}
		}
	}
	return best;
}

export function measureLesionWithConnector(
	imageData: ImageData,
	lesionAreaPixels: number,
): ConnectorDetection & { lesionDiameterMm: number | null } {
	const det = detectConnector(imageData);
	if (!det.detected || det.pixelsPerMm <= 0) return { ...det, lesionDiameterMm: null };
	const diam = (2 * Math.sqrt(lesionAreaPixels / Math.PI)) / det.pixelsPerMm;
	const rounded = Math.round(diam * 10) / 10;
	return { ...det, diameterMm: rounded, lesionDiameterMm: rounded };
}
