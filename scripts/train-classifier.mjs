#!/usr/bin/env node
/**
 * DrAgnes HAM10000 Training Pipeline
 *
 * Downloads the HAM10000 dataset from HuggingFace, extracts dermoscopic
 * features from each image using the same analysis pipeline as the browser
 * classifier, trains a multinomial logistic regression model, evaluates on
 * a held-out 15% stratified test set, and exports the trained weights to
 * a JSON file that the browser-side classifier can consume.
 *
 * Usage:
 *   node scripts/train-classifier.mjs quick   # 100 images per class (~700 total)
 *   node scripts/train-classifier.mjs full    # All 10K images
 *
 * Dependencies (install before running):
 *   npm install --save-dev sharp
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline as streamPipeline } from "node:stream/promises";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, "..");

// ============================================================
// Configuration
// ============================================================

const MODE = process.argv[2] || "quick";
const IMAGES_PER_CLASS_QUICK = 100;
const TEST_RATIO = 0.15;
const DATASET_ID = "hawking32/ham10000_ttv";
const CACHE_DIR = path.join(PROJECT_ROOT, ".cache", "ham10000");
const OUTPUT_PATH = path.join(
	PROJECT_ROOT,
	"src",
	"lib",
	"dragnes",
	"trained-weights-empirical.json",
);

const CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const CLASS_LABELS = {
	akiec: "Actinic Keratosis / Intraepithelial Carcinoma",
	bcc: "Basal Cell Carcinoma",
	bkl: "Benign Keratosis-like Lesion",
	df: "Dermatofibroma",
	mel: "Melanoma",
	nv: "Melanocytic Nevus",
	vasc: "Vascular Lesion",
};

// Feature vector ordering -- must match trained-weights.ts FEATURE_NAMES
const FEATURE_NAMES = [
	"asymmetry",
	"borderScore",
	"colorCount",
	"diameterMm",
	"hasWhite",
	"hasRed",
	"hasLightBrown",
	"hasDarkBrown",
	"hasBlueGray",
	"hasBlack",
	"contrast",
	"homogeneity",
	"entropy",
	"correlation",
	"hasIrregularNetwork",
	"hasIrregularGlobules",
	"hasStreaks",
	"hasBlueWhiteVeil",
	"hasRegressionStructures",
	"structuralScore",
];

const NUM_FEATURES = FEATURE_NAMES.length;
const NUM_CLASSES = CLASS_NAMES.length;

// ============================================================
// Polyfill: ImageData for Node.js
// ============================================================

if (typeof globalThis.ImageData === "undefined") {
	globalThis.ImageData = class ImageData {
		constructor(data, width, height) {
			if (data instanceof Uint8ClampedArray) {
				this.data = data;
			} else if (data instanceof Uint8Array) {
				this.data = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
			} else {
				this.data = new Uint8ClampedArray(data);
			}
			this.width = width;
			this.height = height ?? Math.floor(this.data.length / (4 * width));
		}
	};
}

// ============================================================
// Image loading via sharp
// ============================================================

let sharp;
try {
	sharp = (await import("sharp")).default;
} catch {
	console.error(
		"ERROR: 'sharp' is required but not installed.\n" +
			"Install it with: npm install --save-dev sharp\n",
	);
	process.exit(1);
}

/**
 * Load an image file and return an ImageData-compatible object.
 * Resizes to a fixed 224x224 for consistent feature extraction.
 */
async function loadImage(filePath) {
	const img = sharp(filePath).resize(224, 224, { fit: "fill" }).ensureAlpha();
	const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
	return new ImageData(new Uint8ClampedArray(data), info.width, info.height);
}

// ============================================================
// Feature extraction (mirrors image-analysis.ts in pure JS)
// ============================================================

// --- RGB to LAB ---
function rgbToLab(r, g, b) {
	let rn = r / 255;
	let gn = g / 255;
	let bn = b / 255;

	rn = rn > 0.04045 ? Math.pow((rn + 0.055) / 1.055, 2.4) : rn / 12.92;
	gn = gn > 0.04045 ? Math.pow((gn + 0.055) / 1.055, 2.4) : gn / 12.92;
	bn = bn > 0.04045 ? Math.pow((bn + 0.055) / 1.055, 2.4) : bn / 12.92;

	let x = (rn * 0.4124564 + gn * 0.3575761 + bn * 0.1804375) / 0.95047;
	let y = rn * 0.2126729 + gn * 0.7151522 + bn * 0.072175;
	let z = (rn * 0.0193339 + gn * 0.119192 + bn * 0.9503041) / 1.08883;

	const epsilon = 0.008856;
	const kappa = 903.3;
	x = x > epsilon ? Math.cbrt(x) : (kappa * x + 16) / 116;
	y = y > epsilon ? Math.cbrt(y) : (kappa * y + 16) / 116;
	z = z > epsilon ? Math.cbrt(z) : (kappa * z + 16) / 116;

	return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

// --- Grayscale ---
function toGrayscale(imageData) {
	const { data, width, height } = imageData;
	const gray = new Uint8Array(width * height);
	for (let i = 0; i < gray.length; i++) {
		const px = i * 4;
		gray[i] = Math.round(0.299 * data[px] + 0.587 * data[px + 1] + 0.114 * data[px + 2]);
	}
	return gray;
}

// --- Otsu threshold ---
function otsuThreshold(values, count) {
	const histogram = new Int32Array(256);
	for (let i = 0; i < count; i++) {
		const v = Math.max(0, Math.min(255, Math.round(Number(values[i]))));
		histogram[v]++;
	}
	let sumAll = 0;
	for (let i = 0; i < 256; i++) sumAll += i * histogram[i];

	let sumBg = 0, weightBg = 0, maxVariance = 0, bestThreshold = 0;
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

// --- Largest connected component ---
function largestConnectedComponent(mask, width, height) {
	const labels = new Int32Array(width * height);
	let nextLabel = 1;
	const labelSizes = new Map();

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const idx = y * width + x;
			if (mask[idx] !== 1 || labels[idx] !== 0) continue;
			const label = nextLabel++;
			let size = 0;
			const queue = [idx];
			labels[idx] = label;
			while (queue.length > 0) {
				const cur = queue.pop();
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

	let largestLabel = 0, largestSize = 0;
	for (const [label, size] of labelSizes) {
		if (size > largestSize) { largestSize = size; largestLabel = label; }
	}

	const result = new Uint8Array(width * height);
	for (let i = 0; i < labels.length; i++) {
		result[i] = labels[i] === largestLabel ? 1 : 0;
	}
	return result;
}

// --- Morphological operations ---
function morphDilate(mask, w, h, r) {
	const out = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			let val = 0;
			for (let dy = -r; dy <= r && !val; dy++) {
				for (let dx = -r; dx <= r && !val; dx++) {
					const ny = y + dy, nx = x + dx;
					if (ny >= 0 && ny < h && nx >= 0 && nx < w && mask[ny * w + nx] === 1) val = 1;
				}
			}
			out[y * w + x] = val;
		}
	}
	return out;
}

function morphErode(mask, w, h, r) {
	const out = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			let val = 1;
			for (let dy = -r; dy <= r && val; dy++) {
				for (let dx = -r; dx <= r && val; dx++) {
					const ny = y + dy, nx = x + dx;
					if (ny < 0 || ny >= h || nx < 0 || nx >= w || mask[ny * w + nx] === 0) val = 0;
				}
			}
			out[y * w + x] = val;
		}
	}
	return out;
}

function morphClose(mask, w, h, r) {
	return morphErode(morphDilate(mask, w, h, r), w, h, r);
}

function morphOpen(mask, w, h, r) {
	return morphDilate(morphErode(mask, w, h, r), w, h, r);
}

// --- Segmentation ---
function segmentLesion(imageData) {
	const { data, width, height } = imageData;
	const totalPixels = width * height;

	const lChannel = new Uint8Array(totalPixels);
	for (let i = 0; i < totalPixels; i++) {
		const px = i * 4;
		const [L] = rgbToLab(data[px], data[px + 1], data[px + 2]);
		lChannel[i] = Math.max(0, Math.min(255, Math.round(L * 2.55)));
	}

	const threshold = otsuThreshold(lChannel, totalPixels);
	const initialMask = new Uint8Array(totalPixels);
	for (let i = 0; i < totalPixels; i++) {
		initialMask[i] = lChannel[i] <= threshold ? 1 : 0;
	}

	const closedMask = morphClose(initialMask, width, height, 3);
	const cleanedMask = morphOpen(closedMask, width, height, 2);
	const mask = largestConnectedComponent(cleanedMask, width, height);

	let minX = width, minY = height, maxX = 0, maxY = 0;
	let area = 0, perimeter = 0;

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const idx = y * width + x;
			if (mask[idx] !== 1) continue;
			area++;
			if (x < minX) minX = x;
			if (x > maxX) maxX = x;
			if (y < minY) minY = y;
			if (y > maxY) maxY = y;
			let isBorder = false;
			for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
				const nx = x + dx, ny = y + dy;
				if (nx < 0 || nx >= width || ny < 0 || ny >= height || mask[ny * width + nx] === 0) {
					isBorder = true;
					break;
				}
			}
			if (isBorder) perimeter++;
		}
	}

	if (area < totalPixels * 0.01) {
		return fallbackSegmentation(width, height);
	}

	return {
		mask,
		bbox: { x: minX, y: minY, w: Math.max(1, maxX - minX + 1), h: Math.max(1, maxY - minY + 1) },
		area,
		perimeter,
	};
}

function fallbackSegmentation(width, height) {
	const cx = width / 2, cy = height / 2;
	const rx = width * 0.35, ry = height * 0.35;
	const mask = new Uint8Array(width * height);
	let area = 0, perimeter = 0;
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const dx = (x - cx) / rx, dy = (y - cy) / ry;
			const dist = dx * dx + dy * dy;
			if (dist <= 1.0) {
				mask[y * width + x] = 1;
				area++;
				if (dist > 0.85) perimeter++;
			}
		}
	}
	const bx = Math.max(0, Math.round(cx - rx));
	const by = Math.max(0, Math.round(cy - ry));
	return {
		mask,
		bbox: { x: bx, y: by, w: Math.min(width - bx, Math.round(rx * 2)), h: Math.min(height - by, Math.round(ry * 2)) },
		area,
		perimeter,
	};
}

// --- Asymmetry ---
function measureAsymmetry(mask, width, height, bbox) {
	if (bbox.w === 0 || bbox.h === 0) return 0;
	let sumX = 0, sumY = 0, count = 0;
	for (let y = bbox.y; y < bbox.y + bbox.h; y++) {
		for (let x = bbox.x; x < bbox.x + bbox.w; x++) {
			if (mask[y * width + x] === 1) { sumX += x; sumY += y; count++; }
		}
	}
	if (count === 0) return 0;
	const cx = sumX / count, cy = sumY / count;
	let mxx = 0, myy = 0, mxy = 0;
	for (let y = bbox.y; y < bbox.y + bbox.h; y++) {
		for (let x = bbox.x; x < bbox.x + bbox.w; x++) {
			if (mask[y * width + x] === 1) {
				const dx = x - cx, dy = y - cy;
				mxx += dx * dx; myy += dy * dy; mxy += dx * dy;
			}
		}
	}
	const theta = 0.5 * Math.atan2(2 * mxy, mxx - myy);
	const a1 = measureAxisAsymmetry(mask, width, height, bbox, cx, cy, theta);
	const a2 = measureAxisAsymmetry(mask, width, height, bbox, cx, cy, theta + Math.PI / 2);
	const threshold = 0.15;
	let score = 0;
	if (a1 > threshold) score++;
	if (a2 > threshold) score++;
	return score;
}

function measureAxisAsymmetry(mask, width, height, bbox, cx, cy, theta) {
	const cosT = Math.cos(theta), sinT = Math.sin(theta);
	let mismatch = 0, total = 0;
	for (let y = bbox.y; y < bbox.y + bbox.h; y++) {
		for (let x = bbox.x; x < bbox.x + bbox.w; x++) {
			const dx = x - cx, dy = y - cy;
			const perpDist = -dx * sinT + dy * cosT;
			if (perpDist < 0) continue;
			const mirrorX = Math.round(x + 2 * perpDist * sinT);
			const mirrorY = Math.round(y - 2 * perpDist * cosT);
			if (mirrorX < 0 || mirrorX >= width || mirrorY < 0 || mirrorY >= height) {
				if (mask[y * width + x] === 1) { total++; mismatch++; }
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

// --- Border analysis ---
function analyzeBorder(imageData, mask, width, height) {
	const { data } = imageData;
	const borderPixels = [];
	let cxSum = 0, cySum = 0, lesionCount = 0;
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			if (mask[y * width + x] === 1) { cxSum += x; cySum += y; lesionCount++; }
		}
	}
	if (lesionCount === 0) return 0;
	const cx = cxSum / lesionCount, cy = cySum / lesionCount;
	for (let y = 1; y < height - 1; y++) {
		for (let x = 1; x < width - 1; x++) {
			if (mask[y * width + x] !== 1) continue;
			let isBorder = false;
			for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
				if (mask[(y + dy) * width + (x + dx)] === 0) { isBorder = true; break; }
			}
			if (isBorder) {
				const angle = Math.atan2(y - cy, x - cx);
				const radius = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
				borderPixels.push({ x, y, angle, radius });
			}
		}
	}
	if (borderPixels.length < 16) return 0;
	const octants = Array.from({ length: 8 }, () => ({ radii: [], pixelIndices: [] }));
	for (let i = 0; i < borderPixels.length; i++) {
		const bp = borderPixels[i];
		const normalizedAngle = bp.angle + Math.PI;
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
		let gradientSum = 0, gradientCount = 0;
		for (const pi of oct.pixelIndices) {
			const bp = borderPixels[pi];
			const px = bp.y * width + bp.x;
			for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
				const nx = bp.x + dx, ny = bp.y + dy;
				if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
				if (mask[ny * width + nx] === 0) {
					const lesionPx = px * 4, bgPx = (ny * width + nx) * 4;
					const dr = data[lesionPx] - data[bgPx];
					const dg = data[lesionPx + 1] - data[bgPx + 1];
					const db = data[lesionPx + 2] - data[bgPx + 2];
					gradientSum += Math.sqrt(dr * dr + dg * dg + db * db);
					gradientCount++;
				}
			}
		}
		const avgGradient = gradientCount > 0 ? gradientSum / gradientCount : 0;
		if (cv > 0.25 || avgGradient > 80) irregularCount++;
	}
	return irregularCount;
}

// --- Color analysis ---
const DERM_COLORS = [
	{ name: "white", lab: [95, 0, 0], rgb: [240, 240, 240] },
	{ name: "red", lab: [50, 55, 35], rgb: [190, 60, 50] },
	{ name: "light-brown", lab: [55, 15, 30], rgb: [170, 120, 70] },
	{ name: "dark-brown", lab: [30, 15, 20], rgb: [90, 50, 30] },
	{ name: "blue-gray", lab: [55, -5, -15], rgb: [120, 140, 165] },
	{ name: "black", lab: [10, 0, 0], rgb: [25, 25, 25] },
];

function kMeansLab(pixels, k, maxIter) {
	const n = pixels.length;
	const assignments = new Int32Array(n);
	const centroids = [];
	centroids.push([...pixels[Math.floor(Math.random() * n)]]);

	for (let c = 1; c < k; c++) {
		const dists = new Float64Array(n);
		let totalDist = 0;
		for (let i = 0; i < n; i++) {
			let minDist = Infinity;
			for (const cent of centroids) {
				const d = (pixels[i][0] - cent[0]) ** 2 + (pixels[i][1] - cent[1]) ** 2 + (pixels[i][2] - cent[2]) ** 2;
				if (d < minDist) minDist = d;
			}
			dists[i] = minDist;
			totalDist += minDist;
		}
		let target = Math.random() * totalDist;
		let chosen = 0;
		for (let i = 0; i < n; i++) {
			target -= dists[i];
			if (target <= 0) { chosen = i; break; }
		}
		centroids.push([...pixels[chosen]]);
	}

	for (let iter = 0; iter < maxIter; iter++) {
		let changed = false;
		for (let i = 0; i < n; i++) {
			let bestC = 0, bestDist = Infinity;
			for (let c = 0; c < k; c++) {
				const d = (pixels[i][0] - centroids[c][0]) ** 2 + (pixels[i][1] - centroids[c][1]) ** 2 + (pixels[i][2] - centroids[c][2]) ** 2;
				if (d < bestDist) { bestDist = d; bestC = c; }
			}
			if (assignments[i] !== bestC) { assignments[i] = bestC; changed = true; }
		}
		if (!changed) break;

		const sums = Array.from({ length: k }, () => [0, 0, 0]);
		const counts = new Int32Array(k);
		for (let i = 0; i < n; i++) {
			const c = assignments[i];
			sums[c][0] += pixels[i][0]; sums[c][1] += pixels[i][1]; sums[c][2] += pixels[i][2];
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

function analyzeColors(imageData, mask) {
	const { data, width, height } = imageData;
	const labPixels = [];
	const rgbPixels = [];
	const step = Math.max(1, Math.floor(width * height / 5000));

	for (let i = 0; i < width * height; i += step) {
		if (mask[i] !== 1) continue;
		const px = i * 4;
		const r = data[px], g = data[px + 1], b = data[px + 2];
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

	const k = Math.min(6, labPixels.length);
	const assignments = kMeansLab(labPixels, k, 15);

	const clusterSums = Array.from({ length: k }, () => [0, 0, 0]);
	const clusterRgbSums = Array.from({ length: k }, () => [0, 0, 0]);
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

	const colorPresence = new Map();
	const totalSampled = labPixels.length;

	for (let c = 0; c < k; c++) {
		if (clusterCounts[c] === 0) continue;
		const centroidL = clusterSums[c][0] / clusterCounts[c];
		const centroidA = clusterSums[c][1] / clusterCounts[c];
		const centroidB = clusterSums[c][2] / clusterCounts[c];
		const centroidR = Math.round(clusterRgbSums[c][0] / clusterCounts[c]);
		const centroidG = Math.round(clusterRgbSums[c][1] / clusterCounts[c]);
		const centroidBv = Math.round(clusterRgbSums[c][2] / clusterCounts[c]);

		let bestDist = Infinity, bestColor = DERM_COLORS[0];
		for (const dc of DERM_COLORS) {
			const dL = centroidL - dc.lab[0];
			const dA = centroidA - dc.lab[1];
			const dB = centroidB - dc.lab[2];
			const dist = dL * dL + dA * dA + dB * dB;
			if (dist < bestDist) { bestDist = dist; bestColor = dc; }
		}

		if (bestDist < 1600) {
			const existing = colorPresence.get(bestColor.name);
			if (existing) {
				existing.count += clusterCounts[c];
			} else {
				colorPresence.set(bestColor.name, {
					count: clusterCounts[c],
					rgb: [centroidR, centroidG, centroidBv],
				});
			}
		}
	}

	const minThreshold = totalSampled * 0.03;
	const dominantColors = [];
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

	const hasBlueGray = colorPresence.has("blue-gray") && (colorPresence.get("blue-gray").count / totalSampled) > 0.05;
	const hasWhite = colorPresence.has("white") && (colorPresence.get("white").count / totalSampled) > 0.03;

	return {
		colorCount: Math.max(1, dominantColors.length),
		dominantColors: dominantColors.length > 0 ? dominantColors : [{ name: "light-brown", percentage: 100, rgb: [170, 120, 70] }],
		hasBlueWhiteStructures: hasBlueGray && hasWhite,
	};
}

// --- Texture analysis (GLCM) ---
function analyzeTexture(imageData, mask) {
	const { data, width, height } = imageData;
	const levels = 32;
	const quantized = new Uint8Array(width * height);
	for (let i = 0; i < width * height; i++) {
		if (mask[i] !== 1) continue;
		const px = i * 4;
		const gray = 0.299 * data[px] + 0.587 * data[px + 1] + 0.114 * data[px + 2];
		quantized[i] = Math.min(levels - 1, Math.floor((gray / 256) * levels));
	}

	const directions = [[1, 0], [1, 1], [0, 1], [-1, 1]];
	const glcm = new Float64Array(levels * levels);
	let totalPairs = 0;

	for (const [ddx, ddy] of directions) {
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const idx = y * width + x;
				if (mask[idx] !== 1) continue;
				const nx = x + ddx, ny = y + ddy;
				if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
				const nIdx = ny * width + nx;
				if (mask[nIdx] !== 1) continue;
				const i = quantized[idx], j = quantized[nIdx];
				glcm[i * levels + j]++;
				glcm[j * levels + i]++;
				totalPairs += 2;
			}
		}
	}

	if (totalPairs > 0) {
		for (let i = 0; i < glcm.length; i++) glcm[i] /= totalPairs;
	}

	const muI = new Float64Array(levels), muJ = new Float64Array(levels);
	for (let i = 0; i < levels; i++) {
		for (let j = 0; j < levels; j++) {
			const p = glcm[i * levels + j];
			muI[i] += p; muJ[j] += p;
		}
	}

	let meanI = 0, meanJ = 0;
	for (let k = 0; k < levels; k++) { meanI += k * muI[k]; meanJ += k * muJ[k]; }

	let stdI = 0, stdJ = 0;
	for (let k = 0; k < levels; k++) { stdI += (k - meanI) ** 2 * muI[k]; stdJ += (k - meanJ) ** 2 * muJ[k]; }
	stdI = Math.sqrt(stdI);
	stdJ = Math.sqrt(stdJ);

	let contrast = 0, homogeneity = 0, entropy = 0, correlation = 0;
	for (let i = 0; i < levels; i++) {
		for (let j = 0; j < levels; j++) {
			const p = glcm[i * levels + j];
			if (p === 0) continue;
			contrast += (i - j) ** 2 * p;
			homogeneity += p / (1 + (i - j) ** 2);
			entropy -= p * Math.log2(p + 1e-10);
			if (stdI > 0 && stdJ > 0) correlation += ((i - meanI) * (j - meanJ) * p) / (stdI * stdJ);
		}
	}

	const maxContrast = (levels - 1) ** 2;
	return {
		contrast: Math.min(1, contrast / maxContrast),
		homogeneity,
		entropy: entropy / Math.log2(levels * levels),
		correlation: (correlation + 1) / 2,
	};
}

// --- Structure detection ---
function detectStructures(imageData, mask) {
	const { data, width, height } = imageData;
	const gray = toGrayscale(imageData);

	// LBP
	const lbpHistogram = new Float64Array(256);
	let lbpCount = 0;
	for (let y = 1; y < height - 1; y++) {
		for (let x = 1; x < width - 1; x++) {
			if (mask[y * width + x] !== 1) continue;
			const center = gray[y * width + x];
			let pattern = 0;
			const neighbors = [
				gray[(y - 1) * width + (x - 1)], gray[(y - 1) * width + x],
				gray[(y - 1) * width + (x + 1)], gray[y * width + (x + 1)],
				gray[(y + 1) * width + (x + 1)], gray[(y + 1) * width + x],
				gray[(y + 1) * width + (x - 1)], gray[y * width + (x - 1)],
			];
			for (let n = 0; n < 8; n++) {
				if (neighbors[n] >= center) pattern |= (1 << n);
			}
			lbpHistogram[pattern]++;
			lbpCount++;
		}
	}
	if (lbpCount > 0) {
		for (let i = 0; i < 256; i++) lbpHistogram[i] /= lbpCount;
	}

	let uniformCount = 0, nonUniformCount = 0;
	for (let i = 0; i < 256; i++) {
		if (lbpHistogram[i] < 0.001) continue;
		let transitions = 0;
		for (let bit = 0; bit < 8; bit++) {
			const curr = (i >> bit) & 1;
			const next = (i >> ((bit + 1) % 8)) & 1;
			if (curr !== next) transitions++;
		}
		if (transitions <= 2) uniformCount += lbpHistogram[i];
		else nonUniformCount += lbpHistogram[i];
	}
	const hasIrregularNetwork = nonUniformCount > 0.35;

	// Globules
	let globuleCount = 0, irregularGlobules = 0;
	const radius = 4;
	for (let y = radius; y < height - radius; y += radius) {
		for (let x = radius; x < width - radius; x += radius) {
			if (mask[y * width + x] !== 1) continue;
			const centerGray = gray[y * width + x];
			let isLocalMin = true, neighborSum = 0, neighborCount = 0;
			let minNeighbor = 255, maxNeighbor = 0;
			for (let dy = -radius; dy <= radius; dy++) {
				for (let dx = -radius; dx <= radius; dx++) {
					if (dx === 0 && dy === 0) continue;
					const nx = x + dx, ny = y + dy;
					if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
					if (mask[ny * width + nx] !== 1) continue;
					const nGray = gray[ny * width + nx];
					neighborSum += nGray; neighborCount++;
					if (nGray < minNeighbor) minNeighbor = nGray;
					if (nGray > maxNeighbor) maxNeighbor = nGray;
					if (nGray <= centerGray) isLocalMin = false;
				}
			}
			if (neighborCount > 0 && !isLocalMin) continue;
			const avgNeighbor = neighborCount > 0 ? neighborSum / neighborCount : centerGray;
			if (centerGray < avgNeighbor - 20) {
				globuleCount++;
				if (maxNeighbor - minNeighbor > 60) irregularGlobules++;
			}
		}
	}
	const hasIrregularGlobules = irregularGlobules > 3 && (globuleCount > 0 ? irregularGlobules / globuleCount > 0.4 : false);

	// Streaks
	let minX = width, minY = height, maxX = 0, maxY = 0;
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			if (mask[y * width + x] === 1) {
				if (x < minX) minX = x; if (x > maxX) maxX = x;
				if (y < minY) minY = y; if (y > maxY) maxY = y;
			}
		}
	}
	const bboxW = maxX - minX + 1, bboxH = maxY - minY + 1;
	const borderThickness = Math.max(3, Math.round(Math.min(bboxW, bboxH) * 0.2));
	let streakScore = 0, streakSamples = 0;
	for (let y = minY; y <= maxY; y += 2) {
		for (let x = minX; x <= maxX; x += 2) {
			if (mask[y * width + x] !== 1) continue;
			const distToEdgeX = Math.min(x - minX, maxX - x);
			const distToEdgeY = Math.min(y - minY, maxY - y);
			if (Math.min(distToEdgeX, distToEdgeY) > borderThickness) continue;
			if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) continue;
			const gx =
				-gray[(y - 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)] +
				-2 * gray[y * width + (x - 1)] + 2 * gray[y * width + (x + 1)] +
				-gray[(y + 1) * width + (x - 1)] + gray[(y + 1) * width + (x + 1)];
			const gy =
				-gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)] +
				gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];
			const magnitude = Math.sqrt(gx * gx + gy * gy);
			const scx = (minX + maxX) / 2, scy = (minY + maxY) / 2;
			const radialDx = x - scx, radialDy = y - scy;
			const radialMag = Math.sqrt(radialDx * radialDx + radialDy * radialDy);
			if (radialMag > 0 && magnitude > 30) {
				const gradAngle = Math.atan2(gy, gx);
				const radialAngle = Math.atan2(radialDy, radialDx);
				const angleDiff = Math.abs(Math.sin(gradAngle - radialAngle));
				if (angleDiff < 0.4) streakScore += magnitude;
				streakSamples++;
			}
		}
	}
	const avgStreakScore = streakSamples > 0 ? streakScore / streakSamples : 0;
	const hasStreaks = avgStreakScore > 40;

	// Blue-white veil
	let blueWhiteCount = 0, lesionPixelCount = 0;
	for (let i = 0; i < width * height; i++) {
		if (mask[i] !== 1) continue;
		lesionPixelCount++;
		const px = i * 4;
		const r = data[px], g = data[px + 1], b = data[px + 2];
		const brightness = (r + g + b) / 3;
		if (b > r && b > g && brightness > 80 && brightness < 200 && (b - r) > 20) blueWhiteCount++;
	}
	const hasBlueWhiteVeil = lesionPixelCount > 0 && (blueWhiteCount / lesionPixelCount) > 0.1;

	// Regression structures
	let regressionCount = 0;
	for (let i = 0; i < width * height; i++) {
		if (mask[i] !== 1) continue;
		const px = i * 4;
		const r = data[px], g = data[px + 1], b = data[px + 2];
		const maxC = Math.max(r, g, b), minC = Math.min(r, g, b);
		const saturation = maxC > 0 ? (maxC - minC) / maxC : 0;
		const brightness = (r + g + b) / 3;
		if (brightness > 180 && saturation < 0.15) regressionCount++;
	}
	const hasRegressionStructures = lesionPixelCount > 0 && (regressionCount / lesionPixelCount) > 0.08;

	// Overall structural score
	let structuralScore = 0;
	if (hasIrregularNetwork) structuralScore += 0.2;
	if (hasIrregularGlobules) structuralScore += 0.15;
	if (hasStreaks) structuralScore += 0.25;
	if (hasBlueWhiteVeil) structuralScore += 0.25;
	if (hasRegressionStructures) structuralScore += 0.15;

	return {
		hasIrregularNetwork,
		hasIrregularGlobules,
		hasStreaks,
		hasBlueWhiteVeil,
		hasRegressionStructures,
		structuralScore,
	};
}

// --- Diameter estimation ---
function estimateDiameterMm(areaPixels, imageWidth, magnification = 10) {
	const fieldOfViewMm = 25 / (magnification / 10);
	const pxPerMm = imageWidth / fieldOfViewMm;
	const radiusPx = Math.sqrt(areaPixels / Math.PI);
	const diameterMm = (2 * radiusPx) / pxPerMm;
	return Math.round(diameterMm * 10) / 10;
}

// --- Extract full 20-feature vector ---
function extractFeatureVector(seg, asymmetry, borderScore, colorAnalysis, texture, structures, diameterMm) {
	const colorNames = new Set(colorAnalysis.dominantColors.map((c) => c.name));
	const estimatedDiameterMm = diameterMm ?? (2 * Math.sqrt(seg.area / Math.PI)) / 40;
	return [
		asymmetry,
		borderScore,
		colorAnalysis.colorCount,
		estimatedDiameterMm,
		colorNames.has("white") ? 1 : 0,
		colorNames.has("red") ? 1 : 0,
		colorNames.has("light-brown") ? 1 : 0,
		colorNames.has("dark-brown") ? 1 : 0,
		colorNames.has("blue-gray") ? 1 : 0,
		colorNames.has("black") ? 1 : 0,
		texture.contrast,
		texture.homogeneity,
		texture.entropy,
		texture.correlation,
		structures.hasIrregularNetwork ? 1 : 0,
		structures.hasIrregularGlobules ? 1 : 0,
		structures.hasStreaks ? 1 : 0,
		structures.hasBlueWhiteVeil ? 1 : 0,
		structures.hasRegressionStructures ? 1 : 0,
		structures.structuralScore,
	];
}

/**
 * Run the full feature extraction pipeline on a single image.
 *
 * @param {ImageData} imageData - 224x224 RGBA image
 * @returns {number[]} 20-element feature vector
 */
function extractAllFeatures(imageData) {
	const { width, height } = imageData;

	const seg = segmentLesion(imageData);
	const asymmetry = measureAsymmetry(seg.mask, width, height, seg.bbox);
	const borderScore = analyzeBorder(imageData, seg.mask, width, height);
	const colorAnalysis = analyzeColors(imageData, seg.mask);
	const texture = analyzeTexture(imageData, seg.mask);
	const structures = detectStructures(imageData, seg.mask);
	const diameterMm = estimateDiameterMm(seg.area, width);

	return extractFeatureVector(seg, asymmetry, borderScore, colorAnalysis, texture, structures, diameterMm);
}

// ============================================================
// Dataset download from HuggingFace
// ============================================================

/**
 * Fetch JSON from a URL with basic error handling and retries.
 */
async function fetchJson(url, retries = 3) {
	for (let attempt = 1; attempt <= retries; attempt++) {
		try {
			const res = await fetch(url);
			if (!res.ok) {
				throw new Error(`HTTP ${res.status}: ${res.statusText} (${url})`);
			}
			return await res.json();
		} catch (err) {
			if (attempt === retries) throw err;
			console.warn(`  Retry ${attempt}/${retries} for ${url}: ${err.message}`);
			await new Promise((r) => setTimeout(r, 1000 * attempt));
		}
	}
}

/**
 * Download a file from a URL to a local path, skipping if it already exists.
 */
async function downloadFile(url, destPath, retries = 3) {
	if (fs.existsSync(destPath)) return;

	const dir = path.dirname(destPath);
	fs.mkdirSync(dir, { recursive: true });

	for (let attempt = 1; attempt <= retries; attempt++) {
		try {
			const res = await fetch(url);
			if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
			const fileStream = fs.createWriteStream(destPath);
			// Convert web ReadableStream to Node stream for piping
			const { Readable } = await import("node:stream");
			const nodeStream = Readable.fromWeb(res.body);
			await streamPipeline(nodeStream, fileStream);
			return;
		} catch (err) {
			if (fs.existsSync(destPath)) fs.unlinkSync(destPath);
			if (attempt === retries) throw err;
			console.warn(`  Retry ${attempt}/${retries}: ${err.message}`);
			await new Promise((r) => setTimeout(r, 1000 * attempt));
		}
	}
}

/**
 * List files in a HuggingFace dataset repo using the Hub API.
 *
 * @param {string} repoId - e.g. "hawking32/ham10000_ttv"
 * @param {string} pathPrefix - e.g. "data/train/mel"
 * @returns {Promise<Array<{rfilename: string}>>}
 */
async function listHfFiles(repoId, pathPrefix = "") {
	// The HF API endpoint for listing repo tree
	const baseUrl = `https://huggingface.co/api/datasets/${repoId}/tree/main`;
	const url = pathPrefix ? `${baseUrl}/${pathPrefix}` : baseUrl;
	const items = await fetchJson(url);
	return items;
}

/**
 * Download images for all classes from the HuggingFace dataset.
 * Uses the imagefolder structure: data/{split}/{class_name}/*.jpg
 *
 * @param {string} mode - "quick" or "full"
 * @returns {Promise<Array<{filePath: string, label: string, className: string}>>}
 */
async function downloadDataset(mode) {
	console.log(`\nDownloading HAM10000 dataset (mode: ${mode})...`);
	console.log(`Dataset: ${DATASET_ID}`);
	console.log(`Cache: ${CACHE_DIR}\n`);

	const maxPerClass = mode === "quick" ? IMAGES_PER_CLASS_QUICK : Infinity;
	const allImages = [];

	// The dataset uses train/test/validation splits with class subdirectories.
	// We download from the "train" split for maximum data.
	const splits = ["train"];

	for (const split of splits) {
		for (const cls of CLASS_NAMES) {
			const apiPath = `data/${split}/${cls}`;
			process.stdout.write(`  Listing ${split}/${cls}...`);

			let files;
			try {
				files = await listHfFiles(DATASET_ID, apiPath);
			} catch (err) {
				// Some splits may not have all classes, or the path structure may differ.
				// Try alternative path structures.
				console.warn(` not found, trying alternatives...`);
				try {
					files = await listHfFiles(DATASET_ID, `${split}/${cls}`);
				} catch {
					console.warn(` skipping ${split}/${cls} (not found)`);
					continue;
				}
			}

			// Filter to image files only
			const imageFiles = files.filter(
				(f) => f.type === "file" && /\.(jpg|jpeg|png|bmp)$/i.test(f.path || f.rfilename || ""),
			);

			const selected = imageFiles.slice(0, maxPerClass);
			console.log(` ${selected.length}/${imageFiles.length} images`);

			// Download each image
			let downloaded = 0;
			const concurrency = 10;
			for (let i = 0; i < selected.length; i += concurrency) {
				const batch = selected.slice(i, i + concurrency);
				await Promise.all(
					batch.map(async (file) => {
						const rfilename = file.path || file.rfilename;
						const localPath = path.join(CACHE_DIR, rfilename);
						const url = `https://huggingface.co/datasets/${DATASET_ID}/resolve/main/${rfilename}`;
						try {
							await downloadFile(url, localPath);
							allImages.push({ filePath: localPath, label: cls, className: cls });
							downloaded++;
						} catch (err) {
							console.warn(`    FAILED: ${rfilename}: ${err.message}`);
						}
					}),
				);
				process.stdout.write(`\r  Downloading ${split}/${cls}: ${downloaded}/${selected.length}`);
			}
			console.log();
		}
	}

	console.log(`\nTotal images downloaded: ${allImages.length}`);

	if (allImages.length === 0) {
		console.error(
			"\nERROR: No images were downloaded. Possible causes:\n" +
				"  1. Network issue -- check your internet connection\n" +
				"  2. HuggingFace rate limit -- try again later or set HF_TOKEN env var\n" +
				"  3. Dataset structure changed -- check https://huggingface.co/datasets/" + DATASET_ID + "\n",
		);
		process.exit(1);
	}

	return allImages;
}

// ============================================================
// Stratified train/test split
// ============================================================

/**
 * Perform a stratified split ensuring each class has proportional representation.
 *
 * @param {Array<{filePath: string, label: string}>} images
 * @param {number} testRatio - Fraction of data for the test set (default 0.15)
 * @returns {{ train: Array, test: Array }}
 */
function stratifiedSplit(images, testRatio = 0.15) {
	const train = [];
	const test = [];

	// Group by class
	const byClass = {};
	for (const img of images) {
		(byClass[img.label] ||= []).push(img);
	}

	// Deterministic shuffle (seeded with a simple PRNG for reproducibility)
	let seed = 42;
	function seededRandom() {
		seed = (seed * 1664525 + 1013904223) & 0xffffffff;
		return (seed >>> 0) / 0xffffffff;
	}

	for (const [cls, items] of Object.entries(byClass)) {
		// Shuffle with seeded PRNG
		for (let i = items.length - 1; i > 0; i--) {
			const j = Math.floor(seededRandom() * (i + 1));
			[items[i], items[j]] = [items[j], items[i]];
		}

		const splitIdx = Math.floor(items.length * (1 - testRatio));
		train.push(...items.slice(0, splitIdx));
		test.push(...items.slice(splitIdx));
	}

	return { train, test };
}

// ============================================================
// Multinomial Logistic Regression Training
// ============================================================

/**
 * Softmax function with numerical stability.
 * @param {number[]} logits
 * @returns {number[]}
 */
function softmax(logits) {
	const maxLogit = Math.max(...logits);
	const exps = logits.map((l) => Math.exp(l - maxLogit));
	const sum = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sum);
}

/**
 * Train a multinomial logistic regression classifier with:
 * - Class-weighted cross-entropy loss (inverse frequency weighting)
 * - L2 regularization
 * - Mini-batch SGD with learning rate decay
 *
 * @param {number[][]} features - Array of feature vectors (N x numFeatures)
 * @param {number[]} labels - Class indices (0 to numClasses-1)
 * @param {object} options
 * @returns {{ W: number[][], b: number[] }}
 */
function trainLogisticRegression(features, labels, options = {}) {
	const {
		lr = 0.01,
		iterations = 500,
		lambda = 0.01,
		batchSize = 32,
		lrDecay = 0.998,
		printEvery = 50,
	} = options;

	const N = features.length;
	const numFeatures = features[0].length;
	const numClasses = NUM_CLASSES;

	// Compute class weights (inverse frequency, normalized)
	const classCounts = new Array(numClasses).fill(0);
	for (const l of labels) classCounts[l]++;
	const totalSamples = labels.length;
	const classWeights = classCounts.map((c) =>
		c > 0 ? totalSamples / (numClasses * c) : 1.0,
	);

	// Cap extreme weights to avoid instability
	const maxWeight = Math.max(...classWeights);
	const cappedWeights = classWeights.map((w) => Math.min(w, maxWeight * 0.5 + 0.5));

	console.log("\nClass weights (inverse frequency):");
	for (let c = 0; c < numClasses; c++) {
		console.log(
			`  ${CLASS_NAMES[c].padEnd(6)}: count=${classCounts[c]}, weight=${cappedWeights[c].toFixed(3)}`,
		);
	}

	// Feature normalization: compute mean and std per feature
	const featureMean = new Array(numFeatures).fill(0);
	const featureStd = new Array(numFeatures).fill(0);

	for (const f of features) {
		for (let j = 0; j < numFeatures; j++) featureMean[j] += f[j];
	}
	for (let j = 0; j < numFeatures; j++) featureMean[j] /= N;

	for (const f of features) {
		for (let j = 0; j < numFeatures; j++) featureStd[j] += (f[j] - featureMean[j]) ** 2;
	}
	for (let j = 0; j < numFeatures; j++) {
		featureStd[j] = Math.sqrt(featureStd[j] / N) || 1.0; // avoid division by zero
	}

	// Normalize features
	const normFeatures = features.map((f) =>
		f.map((v, j) => (v - featureMean[j]) / featureStd[j]),
	);

	// Initialize weights with small random values (He initialization)
	const W = Array.from({ length: numClasses }, () =>
		Array.from({ length: numFeatures }, () => (Math.random() - 0.5) * Math.sqrt(2.0 / numFeatures)),
	);
	const b = new Array(numClasses).fill(0);

	// Set initial bias to log of class prior (helps convergence)
	for (let c = 0; c < numClasses; c++) {
		if (classCounts[c] > 0) {
			b[c] = Math.log(classCounts[c] / totalSamples);
		}
	}

	// Training loop
	let currentLr = lr;
	const indices = Array.from({ length: N }, (_, i) => i);

	console.log(
		`\nTraining: ${iterations} iterations, lr=${lr}, lambda=${lambda}, batch=${batchSize}`,
	);
	console.log(`  Features: ${numFeatures}, Classes: ${numClasses}, Samples: ${N}\n`);

	for (let iter = 0; iter < iterations; iter++) {
		// Shuffle indices each epoch
		if (iter % Math.ceil(N / batchSize) === 0) {
			for (let i = indices.length - 1; i > 0; i--) {
				const j = Math.floor(Math.random() * (i + 1));
				[indices[i], indices[j]] = [indices[j], indices[i]];
			}
		}

		// Select mini-batch
		const batchStart = (iter * batchSize) % N;
		const batchIndices = [];
		for (let i = 0; i < batchSize; i++) {
			batchIndices.push(indices[(batchStart + i) % N]);
		}

		// Forward pass: compute logits and softmax for batch
		let batchLoss = 0;

		// Gradient accumulators
		const gradW = Array.from({ length: numClasses }, () => new Array(numFeatures).fill(0));
		const gradB = new Array(numClasses).fill(0);

		for (const idx of batchIndices) {
			const x = normFeatures[idx];
			const y = labels[idx];

			// Compute logits
			const logits = new Array(numClasses);
			for (let c = 0; c < numClasses; c++) {
				let logit = b[c];
				for (let j = 0; j < numFeatures; j++) {
					logit += W[c][j] * x[j];
				}
				logits[c] = logit;
			}

			// Softmax
			const probs = softmax(logits);

			// Weighted cross-entropy loss
			batchLoss -= cappedWeights[y] * Math.log(probs[y] + 1e-10);

			// Gradient: dL/dW[c][j] = weight * (prob[c] - indicator(c == y)) * x[j]
			for (let c = 0; c < numClasses; c++) {
				const error = cappedWeights[y] * (probs[c] - (c === y ? 1 : 0));
				for (let j = 0; j < numFeatures; j++) {
					gradW[c][j] += error * x[j];
				}
				gradB[c] += error;
			}
		}

		// Average gradients over batch
		const batchLen = batchIndices.length;
		for (let c = 0; c < numClasses; c++) {
			for (let j = 0; j < numFeatures; j++) {
				gradW[c][j] /= batchLen;
				// Add L2 regularization gradient
				gradW[c][j] += lambda * W[c][j];
			}
			gradB[c] /= batchLen;
		}

		// Update weights
		for (let c = 0; c < numClasses; c++) {
			for (let j = 0; j < numFeatures; j++) {
				W[c][j] -= currentLr * gradW[c][j];
			}
			b[c] -= currentLr * gradB[c];
		}

		// Learning rate decay
		currentLr *= lrDecay;

		// Log progress
		if (iter % printEvery === 0 || iter === iterations - 1) {
			const avgLoss = batchLoss / batchLen;

			// Quick accuracy on last batch
			let correct = 0;
			for (const idx of batchIndices) {
				const x = normFeatures[idx];
				const logits = new Array(numClasses);
				for (let c = 0; c < numClasses; c++) {
					let logit = b[c];
					for (let j = 0; j < numFeatures; j++) logit += W[c][j] * x[j];
					logits[c] = logit;
				}
				const predicted = logits.indexOf(Math.max(...logits));
				if (predicted === labels[idx]) correct++;
			}
			const batchAcc = (correct / batchLen * 100).toFixed(1);

			console.log(
				`  Iter ${String(iter).padStart(4)}/${iterations} | loss=${avgLoss.toFixed(4)} | batch_acc=${batchAcc}% | lr=${currentLr.toFixed(6)}`,
			);
		}
	}

	// De-normalize weights: since we trained on z-scored features,
	// we need to adjust weights and biases for raw feature inputs.
	// For normalized x_norm = (x - mean) / std:
	//   logit = W_norm * x_norm + b_norm
	//         = W_norm * (x - mean) / std + b_norm
	//         = (W_norm / std) * x + (b_norm - sum(W_norm * mean / std))
	const W_raw = Array.from({ length: numClasses }, () => new Array(numFeatures));
	const b_raw = new Array(numClasses);

	for (let c = 0; c < numClasses; c++) {
		let biasAdjust = 0;
		for (let j = 0; j < numFeatures; j++) {
			W_raw[c][j] = W[c][j] / featureStd[j];
			biasAdjust += W[c][j] * featureMean[j] / featureStd[j];
		}
		b_raw[c] = b[c] - biasAdjust;
	}

	return {
		W: W_raw,
		b: b_raw,
		featureMean,
		featureStd,
		classWeights: cappedWeights,
	};
}

// ============================================================
// Evaluation
// ============================================================

/**
 * Predict class probabilities for a feature vector using trained weights.
 *
 * @param {number[][]} W - Weight matrix (numClasses x numFeatures)
 * @param {number[]} b - Bias vector (numClasses)
 * @param {number[]} features - Raw feature vector
 * @returns {number[]} Probabilities for each class
 */
function predict(W, b, features) {
	const numClasses = W.length;
	const logits = new Array(numClasses);
	for (let c = 0; c < numClasses; c++) {
		let logit = b[c];
		for (let j = 0; j < features.length; j++) {
			logit += W[c][j] * features[j];
		}
		logits[c] = logit;
	}
	return softmax(logits);
}

/**
 * Evaluate the trained model on a test set and print a comprehensive report.
 *
 * @param {number[][]} W
 * @param {number[]} b
 * @param {number[][]} testFeatures
 * @param {number[]} testLabels
 * @returns {object} Metrics object for export
 */
function evaluate(W, b, testFeatures, testLabels) {
	const numClasses = CLASS_NAMES.length;
	const confusion = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
	let correct = 0;

	for (let i = 0; i < testFeatures.length; i++) {
		const probs = predict(W, b, testFeatures[i]);
		const predicted = probs.indexOf(Math.max(...probs));
		const actual = testLabels[i];
		confusion[actual][predicted]++;
		if (predicted === actual) correct++;
	}

	const accuracy = correct / testFeatures.length;

	console.log("\n" + "=".repeat(70));
	console.log("  DrAgnes Classifier Evaluation");
	console.log("=".repeat(70));
	console.log(`\n  Test set: ${testFeatures.length} images (${(TEST_RATIO * 100).toFixed(0)}% holdout)`);
	console.log(`  Overall accuracy: ${(accuracy * 100).toFixed(1)}%`);

	// Per-class metrics
	console.log("\n  Per-class metrics:");
	console.log(
		"  " + "Class".padEnd(8) + "Sens".padStart(8) + "Spec".padStart(8) +
		"Prec".padStart(8) + "F1".padStart(8) + "  N_test",
	);
	console.log("  " + "-".repeat(48));

	const metrics = {};
	for (let c = 0; c < numClasses; c++) {
		const tp = confusion[c][c];
		const fn = confusion[c].reduce((a, b) => a + b, 0) - tp;
		const fp = confusion.reduce((a, row) => a + row[c], 0) - tp;
		const tn = testFeatures.length - tp - fn - fp;
		const sensitivity = tp + fn > 0 ? tp / (tp + fn) : 0;
		const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
		const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
		const f1 = precision + sensitivity > 0 ? (2 * precision * sensitivity) / (precision + sensitivity) : 0;
		const nTest = tp + fn;

		console.log(
			"  " +
				CLASS_NAMES[c].padEnd(8) +
				`${(sensitivity * 100).toFixed(1)}%`.padStart(8) +
				`${(specificity * 100).toFixed(1)}%`.padStart(8) +
				`${(precision * 100).toFixed(1)}%`.padStart(8) +
				`${(f1 * 100).toFixed(1)}%`.padStart(8) +
				`  ${nTest}`,
		);

		metrics[CLASS_NAMES[c]] = {
			sensitivity: Math.round(sensitivity * 1000) / 1000,
			specificity: Math.round(specificity * 1000) / 1000,
			precision: Math.round(precision * 1000) / 1000,
			f1: Math.round(f1 * 1000) / 1000,
			nTest,
		};
	}

	// Melanoma sensitivity highlight
	const melSens = metrics.mel?.sensitivity || 0;
	console.log("\n  --- CRITICAL METRIC ---");
	console.log(`  Melanoma sensitivity: ${(melSens * 100).toFixed(1)}%`);
	if (melSens >= 0.9) {
		console.log("  Status: PASS (>= 90% threshold for clinical relevance)");
	} else if (melSens >= 0.8) {
		console.log("  Status: MARGINAL (80-90%, may need tuning)");
	} else {
		console.log("  Status: BELOW THRESHOLD (< 80%, needs improvement)");
	}

	// DermaSensor benchmark comparison
	console.log("\n  DermaSensor comparison benchmarks:");
	console.log("    DermaSensor melanoma sensitivity: 95.5% (FDA submission)");
	console.log("    DermaSensor melanoma specificity: 20-40% (published estimates)");
	console.log(`    DrAgnes melanoma sensitivity: ${(melSens * 100).toFixed(1)}%`);
	console.log(`    DrAgnes melanoma specificity: ${((metrics.mel?.specificity || 0) * 100).toFixed(1)}%`);

	// Confusion matrix
	console.log("\n  Confusion Matrix (rows=actual, cols=predicted):");
	console.log("  " + "".padStart(8) + CLASS_NAMES.map((c) => c.padStart(7)).join(""));
	for (let i = 0; i < numClasses; i++) {
		const row = confusion[i].map((v) => String(v).padStart(7)).join("");
		console.log("  " + CLASS_NAMES[i].padEnd(8) + row);
	}

	console.log("\n" + "=".repeat(70));

	return {
		accuracy: Math.round(accuracy * 1000) / 1000,
		perClass: metrics,
		confusionMatrix: confusion,
	};
}

// ============================================================
// Main pipeline
// ============================================================

async function main() {
	const startTime = Date.now();

	console.log("=".repeat(70));
	console.log("  DrAgnes HAM10000 Training Pipeline");
	console.log("=".repeat(70));
	console.log(`  Mode: ${MODE}`);
	console.log(`  Test ratio: ${TEST_RATIO}`);
	console.log(`  Output: ${OUTPUT_PATH}`);
	console.log(`  Date: ${new Date().toISOString()}`);

	// Step 1: Download dataset
	const allImages = await downloadDataset(MODE);

	// Step 2: Stratified split
	console.log("\nPerforming stratified 85/15 train/test split...");
	const { train, test } = stratifiedSplit(allImages, TEST_RATIO);

	console.log(`  Train set: ${train.length} images`);
	console.log(`  Test set: ${test.length} images`);

	// Show class distribution
	const trainCounts = {};
	const testCounts = {};
	for (const img of train) trainCounts[img.label] = (trainCounts[img.label] || 0) + 1;
	for (const img of test) testCounts[img.label] = (testCounts[img.label] || 0) + 1;

	console.log("\n  Class distribution:");
	console.log("  " + "Class".padEnd(8) + "Train".padStart(8) + "Test".padStart(8));
	for (const cls of CLASS_NAMES) {
		console.log(
			"  " +
				cls.padEnd(8) +
				String(trainCounts[cls] || 0).padStart(8) +
				String(testCounts[cls] || 0).padStart(8),
		);
	}

	// Step 3: Extract features from training set
	console.log("\nExtracting features from training images...");
	const trainFeatures = [];
	const trainLabels = [];
	let featureErrors = 0;

	for (let i = 0; i < train.length; i++) {
		const img = train[i];
		try {
			const imageData = await loadImage(img.filePath);
			const features = extractAllFeatures(imageData);
			trainFeatures.push(features);
			trainLabels.push(CLASS_NAMES.indexOf(img.label));
		} catch (err) {
			featureErrors++;
			if (featureErrors <= 5) {
				console.warn(`  WARNING: Failed to extract features from ${img.filePath}: ${err.message}`);
			}
		}

		if ((i + 1) % 50 === 0 || i === train.length - 1) {
			const pct = ((i + 1) / train.length * 100).toFixed(1);
			const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
			process.stdout.write(
				`\r  Progress: ${i + 1}/${train.length} (${pct}%) | Errors: ${featureErrors} | Elapsed: ${elapsed}s`,
			);
		}
	}
	console.log();

	if (featureErrors > 0) {
		console.log(`  Total feature extraction errors: ${featureErrors}/${train.length}`);
	}

	// Step 4: Extract features from test set
	console.log("\nExtracting features from test images...");
	const testFeatures = [];
	const testLabels = [];

	for (let i = 0; i < test.length; i++) {
		const img = test[i];
		try {
			const imageData = await loadImage(img.filePath);
			const features = extractAllFeatures(imageData);
			testFeatures.push(features);
			testLabels.push(CLASS_NAMES.indexOf(img.label));
		} catch {
			// Skip failed images in test set
		}

		if ((i + 1) % 25 === 0 || i === test.length - 1) {
			process.stdout.write(`\r  Progress: ${i + 1}/${test.length}`);
		}
	}
	console.log();

	// Step 5: Train the model
	console.log("\nTraining multinomial logistic regression...");
	const { W, b, featureMean, featureStd, classWeights } = trainLogisticRegression(
		trainFeatures,
		trainLabels,
		{
			lr: 0.01,
			iterations: 500,
			lambda: 0.01,
			batchSize: 32,
			lrDecay: 0.998,
			printEvery: 50,
		},
	);

	// Step 6: Evaluate on test set
	const metrics = evaluate(W, b, testFeatures, testLabels);

	// Step 7: Export weights
	console.log(`\nExporting trained weights to:\n  ${OUTPUT_PATH}`);

	const output = {
		version: "1.0.0-ham10000-trained",
		trainedOn: `HAM10000 (${DATASET_ID})`,
		mode: MODE,
		splitRatio: `${Math.round((1 - TEST_RATIO) * 100)}/${Math.round(TEST_RATIO * 100)} stratified`,
		trainSize: trainFeatures.length,
		testSize: testFeatures.length,
		trainDate: new Date().toISOString(),
		metrics: {
			accuracy: metrics.accuracy,
			perClass: metrics.perClass,
		},
		featureNames: FEATURE_NAMES,
		classNames: CLASS_NAMES,
		classLabels: CLASS_LABELS,
		// The weight matrix: 7 classes x 20 features
		// These operate on RAW (un-normalized) feature values so the browser
		// classifier can use them directly without knowing mean/std.
		weights: W,
		biases: b,
		// Also include normalization parameters in case someone wants to
		// use the normalized-space weights directly
		normalization: {
			featureMean,
			featureStd,
		},
		classWeights,
		confusionMatrix: metrics.confusionMatrix,
	};

	fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
	fs.writeFileSync(OUTPUT_PATH, JSON.stringify(output, null, 2));
	console.log("  Done.");

	// Summary
	const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
	console.log("\n" + "=".repeat(70));
	console.log("  Pipeline complete");
	console.log("=".repeat(70));
	console.log(`  Total time: ${elapsed}s`);
	console.log(`  Train images: ${trainFeatures.length}`);
	console.log(`  Test images: ${testFeatures.length}`);
	console.log(`  Overall accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`);
	console.log(`  Melanoma sensitivity: ${((metrics.perClass.mel?.sensitivity || 0) * 100).toFixed(1)}%`);
	console.log(`  Output: ${OUTPUT_PATH}`);
	console.log("=".repeat(70) + "\n");
}

main().catch((err) => {
	console.error("\nFATAL ERROR:", err.message);
	console.error(err.stack);
	process.exit(1);
});
