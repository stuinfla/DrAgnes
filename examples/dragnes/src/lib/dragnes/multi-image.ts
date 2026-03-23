/**
 * Quality-weighted multi-image consensus classification.
 * Scores each image's quality, classifies independently, combines via quality-
 * weighted probability averaging. Melanoma safety gate: if ANY single image flags
 * mel > 60%, it stays prominent in consensus (sensitivity over specificity).
 */
import type { ClassificationResult, ClassProbability, LesionClass } from "./types";
import { LESION_LABELS } from "./types";
import { detectLesionPresence } from "./image-analysis";
import { DermClassifier } from "./classifier";

const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const MELANOMA_SAFETY_THRESHOLD = 0.6;

// ── Interfaces ──────────────────────────────────────────────────────────────

export interface ImageQualityScore {
	sharpness: number;        // Laplacian-variance sharpness [0, 1]
	contrast: number;         // RMS contrast of grayscale pixels [0, 1]
	segmentationQuality: number; // Lesion-presence confidence [0, 1]
	overallScore: number;     // 0.4*sharpness + 0.3*contrast + 0.3*segQuality
}

export interface MultiImageResult extends ClassificationResult {
	perImageResults: ClassificationResult[];
	consensusMethod: string;
	agreementScore: number;   // Inter-image agreement [0, 1]
	qualityScores: ImageQualityScore[];
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** RGBA pixel at offset i -> grayscale (Rec. 709 luminance). */
function rgbaToGray(data: Uint8ClampedArray, i: number): number {
	return 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
}

/** Cosine similarity between two equal-length vectors. */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
	let dot = 0, magA = 0, magB = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		magA += a[i] * a[i];
		magB += b[i] * b[i];
	}
	const denom = Math.sqrt(magA) * Math.sqrt(magB);
	return denom > 0 ? dot / denom : 0;
}

/**
 * Average pairwise cosine similarity of probability distributions.
 * Returns 1 when all distributions are identical, approaches 0 when divergent.
 */
function computeAgreement(results: ClassificationResult[]): number {
	if (results.length < 2) return 1;
	const vectors = results.map((r) => {
		const vec = new Float32Array(CLASSES.length);
		for (const p of r.probabilities) {
			const idx = CLASSES.indexOf(p.className);
			if (idx >= 0) vec[idx] = p.probability;
		}
		return vec;
	});
	let simSum = 0, pairs = 0;
	for (let i = 0; i < vectors.length; i++) {
		for (let j = i + 1; j < vectors.length; j++) {
			simSum += cosineSimilarity(vectors[i], vectors[j]);
			pairs++;
		}
	}
	return pairs > 0 ? simSum / pairs : 1;
}

// ── Quality scoring ─────────────────────────────────────────────────────────

/**
 * Score image quality for dermoscopic classification.
 * - Sharpness: Laplacian variance (3x3 kernel [0 1 0 / 1 -4 1 / 0 1 0])
 * - Contrast: RMS contrast of grayscale values
 * - Segmentation quality: lesion-presence detection confidence
 */
export function scoreImageQuality(imageData: ImageData): ImageQualityScore {
	const { data, width, height } = imageData;

	// Grayscale conversion
	const gray = new Float32Array(width * height);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			gray[y * width + x] = rgbaToGray(data, (y * width + x) * 4);
		}
	}

	// Sharpness: Laplacian variance, normalized to [0, 1] (cap at variance 2000)
	let lapSum = 0, lapSumSq = 0, lapCount = 0;
	for (let y = 1; y < height - 1; y++) {
		for (let x = 1; x < width - 1; x++) {
			const lap =
				gray[(y - 1) * width + x] + gray[(y + 1) * width + x] +
				gray[y * width + (x - 1)] + gray[y * width + (x + 1)] -
				4 * gray[y * width + x];
			lapSum += lap;
			lapSumSq += lap * lap;
			lapCount++;
		}
	}
	const lapMean = lapSum / lapCount;
	const sharpness = Math.min(1, Math.max(0, (lapSumSq / lapCount - lapMean * lapMean) / 2000));

	// Contrast: RMS, normalized to [0, 1] (max theoretical RMS = 127.5)
	let graySum = 0;
	for (let i = 0; i < gray.length; i++) graySum += gray[i];
	const grayMean = graySum / gray.length;
	let sqDiffSum = 0;
	for (let i = 0; i < gray.length; i++) sqDiffSum += (gray[i] - grayMean) ** 2;
	const contrast = Math.min(1, Math.max(0, Math.sqrt(sqDiffSum / gray.length) / 127.5));

	// Segmentation quality from lesion presence detection
	const segmentationQuality = detectLesionPresence(imageData).confidence;

	const overallScore = 0.4 * sharpness + 0.3 * contrast + 0.3 * segmentationQuality;
	return { sharpness, contrast, segmentationQuality, overallScore };
}

// ── Multi-image consensus ───────────────────────────────────────────────────

/**
 * Classify a lesion from multiple images using quality-weighted consensus.
 *
 * 1. Classifies each image independently via classifyWithDemographics.
 * 2. Scores each image's quality.
 * 3. Combines per-class probabilities via quality-weighted averaging.
 * 4. Applies melanoma safety gate if any single image flags mel > 60%.
 */
export async function classifyMultiImage(
	classifier: DermClassifier,
	images: ImageData[],
	demographics?: { age?: number; sex?: "male" | "female"; localization?: string },
): Promise<MultiImageResult> {
	if (images.length === 0) throw new Error("classifyMultiImage requires at least one image");

	const startTime = performance.now();
	const perImageResults: ClassificationResult[] = [];
	const qualityScores: ImageQualityScore[] = [];

	for (const img of images) {
		const result = await classifier.classifyWithDemographics(img, demographics);
		perImageResults.push(result);
		qualityScores.push(scoreImageQuality(img));
	}

	// Quality-weighted probability averaging
	const totalWeight = qualityScores.reduce((s, q) => s + q.overallScore, 0);
	const wp: Record<LesionClass, number> = {} as Record<LesionClass, number>;
	for (const cls of CLASSES) wp[cls] = 0;

	for (let i = 0; i < perImageResults.length; i++) {
		const w = totalWeight > 0 ? qualityScores[i].overallScore / totalWeight : 1 / images.length;
		for (const p of perImageResults[i].probabilities) wp[p.className] += p.probability * w;
	}

	// Normalize to sum = 1
	let sum = 0;
	for (const cls of CLASSES) sum += wp[cls];
	if (sum > 0) for (const cls of CLASSES) wp[cls] /= sum;

	// Melanoma safety gate: if any image flagged mel > threshold, preserve it
	const maxMelProb = Math.max(
		...perImageResults.map((r) => r.probabilities.find((p) => p.className === "mel")?.probability ?? 0),
	);
	if (maxMelProb > MELANOMA_SAFETY_THRESHOLD && wp.mel < maxMelProb) {
		const boost = maxMelProb - wp.mel;
		wp.mel = maxMelProb;
		const others = CLASSES.filter((c) => c !== "mel");
		for (const cls of others) wp[cls] = Math.max(0, wp[cls] - boost / others.length);
		// Re-normalize after safety gate adjustment
		let reSum = 0;
		for (const cls of CLASSES) reSum += wp[cls];
		if (reSum > 0) for (const cls of CLASSES) wp[cls] /= reSum;
	}

	// Build sorted output probabilities
	const probabilities: ClassProbability[] = CLASSES.map((cls) => ({
		className: cls,
		probability: wp[cls],
		label: LESION_LABELS[cls],
	})).sort((a, b) => b.probability - a.probability);

	return {
		topClass: probabilities[0].className,
		confidence: probabilities[0].probability,
		probabilities,
		modelId: perImageResults[0].modelId,
		inferenceTimeMs: performance.now() - startTime,
		usedWasm: perImageResults.some((r) => r.usedWasm),
		perImageResults,
		consensusMethod: "quality-weighted-average",
		agreementScore: computeAgreement(perImageResults),
		qualityScores,
	};
}
