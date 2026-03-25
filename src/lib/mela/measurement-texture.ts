/**
 * Skin texture frequency analysis for automatic lesion measurement.
 * Detects pore/ridge spacing via 2D FFT on a skin-only margin patch,
 * then derives a pixels-per-mm calibration factor. ADR-121 Phase 3.
 */
import type { BodyLocation } from "./types";
import { nextPow2, powerSpectrum2d } from "./fft";

/** Expected pore/ridge spacing (mm) by body location. */
export const PORE_SPACING_MM: Record<BodyLocation, number> = {
	head: 0.20, neck: 0.30, trunk: 0.40, upper_extremity: 0.35,
	lower_extremity: 0.40, palms_soles: 0.50, genital: 0.30, unknown: 0.35,
};

export interface TextureMeasurement {
	detected: boolean;
	pixelsPerMm: number;
	confidence: number;        // 0-1
	poreSpacingPx: number;     // detected pore spacing in pixels
	expectedSpacingMm: number; // from body location lookup
	bodyLocation: BodyLocation;
}

const MIN_PEAK_RATIO = 2.5; // minimum peak-to-average for confident detection
const MIN_FREQ_BIN = 3;     // skip DC neighbourhood

/** Extract a grayscale patch from the outer 30% margin (avoids central lesion). */
function extractMarginPatch(img: ImageData, sz: number): Float64Array | null {
	const margin = Math.floor(0.3 * Math.min(img.width, img.height));
	if (margin < sz) return null;
	const patch = new Float64Array(sz * sz);
	const d = img.data;
	for (let py = 0; py < sz; py++) {
		for (let px = 0; px < sz; px++) {
			const i = (py * img.width + px) * 4;
			patch[py * sz + px] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
		}
	}
	return patch;
}

/** Apply 2D Hann window in-place to reduce spectral leakage. */
function applyHann(p: Float64Array, sz: number): void {
	for (let y = 0; y < sz; y++) {
		const wy = 0.5 * (1 - Math.cos(2 * Math.PI * y / (sz - 1)));
		for (let x = 0; x < sz; x++) {
			const wx = 0.5 * (1 - Math.cos(2 * Math.PI * x / (sz - 1)));
			p[y * sz + x] *= wy * wx;
		}
	}
}

/** Radial average of power spectrum: mean power at each integer frequency. */
function radialProfile(ps: Float64Array, sz: number): { freq: number[]; power: number[] } {
	const half = sz >> 1;
	const sums = new Float64Array(half), counts = new Float64Array(half);
	for (let y = 0; y < sz; y++) {
		const fy = y < half ? y : y - sz;
		for (let x = 0; x < sz; x++) {
			const fx = x < half ? x : x - sz;
			const r = Math.round(Math.sqrt(fx * fx + fy * fy));
			if (r > 0 && r < half) { sums[r] += ps[y * sz + x]; counts[r]++; }
		}
	}
	const freq: number[] = [], power: number[] = [];
	for (let r = 1; r < half; r++) {
		if (counts[r] > 0) { freq.push(r); power.push(sums[r] / counts[r]); }
	}
	return { freq, power };
}

/**
 * Measure skin texture to derive a pixels-per-mm calibration.
 * Analyses pore/ridge spacing in the image margin via FFT.
 */
export function measureFromSkinTexture(
	imageData: ImageData, bodyLocation: BodyLocation,
): TextureMeasurement {
	const expectedMm = PORE_SPACING_MM[bodyLocation];
	const noResult: TextureMeasurement = {
		detected: false, pixelsPerMm: 0, confidence: 0,
		poreSpacingPx: 0, expectedSpacingMm: expectedMm, bodyLocation,
	};

	// Choose patch size: 128 preferred, 64 fallback, minimum 32
	const minDim = Math.min(imageData.width, imageData.height);
	const raw = minDim >= 428 ? 128 : minDim >= 214 ? 64 : 0;
	const patchSize = nextPow2(raw || 32);
	if (patchSize < 32) return noResult;

	const patch = extractMarginPatch(imageData, patchSize);
	if (!patch) return noResult;

	applyHann(patch, patchSize);
	const ps = powerSpectrum2d(patch, patchSize, patchSize);
	const { freq, power } = radialProfile(ps, patchSize);
	if (freq.length === 0) return noResult;

	// Find peak spatial frequency (skip DC neighbourhood)
	let peakIdx = -1, peakVal = -Infinity, total = 0, cnt = 0;
	for (let i = 0; i < freq.length; i++) {
		if (freq[i] < MIN_FREQ_BIN) continue;
		total += power[i]; cnt++;
		if (power[i] > peakVal) { peakVal = power[i]; peakIdx = i; }
	}
	if (peakIdx < 0 || cnt === 0) return noResult;

	const avg = total / cnt;
	const ratio = peakVal / avg;
	if (ratio < MIN_PEAK_RATIO) return noResult;

	const poreSpacingPx = patchSize / freq[peakIdx];
	const pixelsPerMm = poreSpacingPx / expectedMm;
	const confidence = Math.min(1, (ratio - MIN_PEAK_RATIO) / (10 - MIN_PEAK_RATIO));

	return {
		detected: true, pixelsPerMm,
		confidence: Math.round(confidence * 100) / 100,
		poreSpacingPx: Math.round(poreSpacingPx * 100) / 100,
		expectedSpacingMm: expectedMm, bodyLocation,
	};
}
