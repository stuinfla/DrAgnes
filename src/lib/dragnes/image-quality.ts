/**
 * Image Quality Gating -- ADR-121 Phase 1
 *
 * Real-time assessment (sharpness, brightness, contrast, framing, glare).
 * Samples every 3rd pixel to stay <50ms on mobile.
 */

export interface QualityCheck {
	name: string;
	passed: boolean;
	score: number;
	message: string;
}

export interface ImageQualityResult {
	grade: "good" | "acceptable" | "poor";
	overallScore: number;
	checks: QualityCheck[];
	suggestion: string | null;
}

const SHARP_POOR = 0.15;
const SHARP_OK = 0.30;
const BRIGHT_LOW = 50 / 255;
const BRIGHT_HIGH = 220 / 255;
const CONTRAST_POOR = 0.08;
const GLARE_THRESH = 245;
const GLARE_PCT = 0.05;
const S = 3; // sampling step

type D = Uint8ClampedArray;
const gray = (d: D, i: number) => (d[i] + d[i + 1] + d[i + 2]) / 3;
const center = (w: number, h: number) => ({
	x0: (w * 0.25) | 0, y0: (h * 0.25) | 0,
	x1: (w * 0.75) | 0, y1: (h * 0.75) | 0,
});

function checkSharpness(d: D, w: number, h: number): QualityCheck {
	let sum = 0, sumSq = 0, n = 0;
	for (let y = S; y < h - S; y += S) {
		for (let x = S; x < w - S; x += S) {
			const c = gray(d, (y * w + x) * 4);
			const lap = (gray(d, ((y - S) * w + x) * 4) + gray(d, ((y + S) * w + x) * 4)
				+ gray(d, (y * w + x - S) * 4) + gray(d, (y * w + x + S) * 4) - 4 * c) / 255;
			sum += lap; sumSq += lap * lap; n++;
		}
	}
	const score = Math.min((n > 0 ? sumSq / n - (sum / n) ** 2 : 0) / 0.04, 1);
	const passed = score >= SHARP_POOR;
	return { name: "sharpness", passed, score,
		message: !passed ? "Image is blurry \u2014 hold steady and tap to focus."
			: score < SHARP_OK ? "Slightly soft \u2014 try holding steadier." : "Sharp" };
}

function checkBrightness(d: D, w: number, h: number): QualityCheck {
	const { x0, y0, x1, y1 } = center(w, h);
	let sum = 0, n = 0;
	for (let y = y0; y < y1; y += S)
		for (let x = x0; x < x1; x += S) { sum += gray(d, (y * w + x) * 4); n++; }
	const mean = n > 0 ? sum / n / 255 : 0.5;
	const score = 1 - 2 * Math.abs(mean - 0.5);
	const tooDark = mean < BRIGHT_LOW, tooBright = mean > BRIGHT_HIGH;
	return { name: "brightness", passed: !tooDark && !tooBright, score,
		message: tooDark ? "Too dark \u2014 move to better lighting."
			: tooBright ? "Too bright \u2014 reduce direct light." : "Brightness OK" };
}

function checkContrast(d: D, w: number, h: number): QualityCheck {
	const { x0, y0, x1, y1 } = center(w, h);
	let sum = 0, sumSq = 0, n = 0;
	for (let y = y0; y < y1; y += S)
		for (let x = x0; x < x1; x += S) {
			const lum = gray(d, (y * w + x) * 4) / 255;
			sum += lum; sumSq += lum * lum; n++;
		}
	const mean = n > 0 ? sum / n : 0;
	const rms = n > 0 ? Math.sqrt(sumSq / n - mean * mean) : 0;
	const score = Math.min(rms / 0.25, 1);
	return { name: "contrast", passed: rms >= CONTRAST_POOR, score,
		message: rms >= CONTRAST_POOR ? "Contrast OK" : "Image looks washed out \u2014 ensure good lighting." };
}

function checkFraming(d: D, w: number, h: number): QualityCheck {
	const { x0, y0, x1, y1 } = center(w, h);
	let cSum = 0, cN = 0, oSum = 0, oN = 0;
	for (let y = 0; y < h; y += S)
		for (let x = 0; x < w; x += S) {
			const lum = gray(d, (y * w + x) * 4);
			if (x >= x0 && x < x1 && y >= y0 && y < y1) { cSum += lum; cN++; }
			else { oSum += lum; oN++; }
		}
	const diff = Math.abs((cN ? cSum / cN : 128) - (oN ? oSum / oN : 128)) / 255;
	const score = Math.min(diff / 0.10, 1);
	return { name: "framing", passed: diff > 0.03, score,
		message: diff > 0.03 ? "Spot centered" : "Move closer to the spot and center it in frame." };
}

function checkGlare(d: D, w: number, h: number): QualityCheck {
	const { x0, y0, x1, y1 } = center(w, h);
	let glare = 0, total = 0;
	for (let y = y0; y < y1; y += S)
		for (let x = x0; x < x1; x += S) {
			const i = (y * w + x) * 4;
			if (d[i] > GLARE_THRESH && d[i + 1] > GLARE_THRESH && d[i + 2] > GLARE_THRESH) glare++;
			total++;
		}
	const ratio = total > 0 ? glare / total : 0;
	const score = Math.max(1 - ratio / GLARE_PCT, 0);
	return { name: "glare", passed: ratio <= GLARE_PCT, score,
		message: ratio <= GLARE_PCT ? "No glare" : "Glare detected \u2014 tilt the phone slightly." };
}

export function assessImageQuality(imageData: ImageData): ImageQualityResult {
	const { data, width: w, height: h } = imageData;
	const checks: QualityCheck[] = [
		checkSharpness(data, w, h),
		checkBrightness(data, w, h),
		checkContrast(data, w, h),
		checkFraming(data, w, h),
		checkGlare(data, w, h),
	];

	const overallScore = checks.reduce((s, c) => s + c.score, 0) / checks.length;
	const failures = checks.filter((c) => !c.passed);
	const hasCritical = failures.some(
		(f) => (f.name === "sharpness" || f.name === "brightness") && f.score < SHARP_POOR,
	);

	const grade: ImageQualityResult["grade"] =
		failures.length === 0 ? "good"
			: hasCritical || failures.length > 2 ? "poor"
				: "acceptable";

	const worst = failures.length > 0
		? failures.reduce((a, b) => (a.score < b.score ? a : b))
		: null;

	return { grade, overallScore, checks, suggestion: worst?.message ?? null };
}
