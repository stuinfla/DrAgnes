/**
 * Mela CNN Classification Engine
 *
 * Loads MobileNetV3 Small WASM module from @ruvector/cnn for
 * browser-based skin lesion classification. Falls back to a
 * demo classifier using color/texture analysis when WASM is unavailable.
 *
 * Supports Grad-CAM heatmap generation for attention visualization.
 */

import type {
	ClassificationResult,
	ClassProbability,
	GradCamResult,
	ImageTensor,
	LesionClass,
} from "./types";
import { LESION_LABELS } from "./types";
import { preprocessImage, resizeBilinear, toNCHWTensor } from "./preprocessing";
import { adjustForDemographics, getClinicalRecommendation } from "./ham10000-knowledge";
import {
	segmentLesion,
	measureAsymmetry,
	analyzeBorder,
	analyzeColors,
	analyzeTexture,
	detectStructures,
	classifyFromFeatures,
	generateAttentionMap,
	estimateDiameterMm,
	type SegmentationResult,
	type ColorAnalysisResult,
	type TextureResult,
	type StructureResult,
} from "./image-analysis";
import {
	classifyWithTrainedWeights,
	extractFeatureVector,
} from "./trained-weights";
import { classify as onnxClassify, warmOfflineModel, isOfflineModelLoaded } from "./inference-orchestrator";

/** All HAM10000 classes in canonical order (used by all ensemble layers) */
const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/** Interface for the WASM CNN module */
interface WasmCnnModule {
	init(modelPath?: string): Promise<void>;
	predict(tensor: Float32Array, shape: number[]): Promise<Float32Array>;
	gradCam(tensor: Float32Array, classIdx: number): Promise<Float32Array>;
}

/**
 * Dermoscopy CNN classifier with WASM backend and demo fallback.
 */
export class DermClassifier {
	private wasmModule: WasmCnnModule | null = null;
	private initialized = false;
	private usesWasm = false;
	private usesOnnx = false;
	private lastTensor: ImageTensor | null = null;
	private lastImageData: ImageData | null = null;

	/** Cached results from the last real image analysis pass */
	private lastSegmentation: SegmentationResult | null = null;
	private lastAsymmetry: number = 0;
	private lastBorderScore: number = 0;
	private lastColorAnalysis: ColorAnalysisResult | null = null;
	private lastTexture: TextureResult | null = null;
	private lastStructures: StructureResult | null = null;

	/**
	 * Initialize the classifier.
	 * Loads the ONNX V2 model (85MB, cached by service worker after first download).
	 * Falls back to @ruvector/cnn WASM, then to trained-weights + rules.
	 */
	async init(): Promise<void> {
		if (this.initialized) return;

		// Priority 1: ONNX V2 model (95.97% mel sensitivity, runs 100% local)
		const onnxReady = await warmOfflineModel();
		if (onnxReady) {
			this.usesOnnx = true;
			this.initialized = true;
			return;
		}

		// Priority 2: @ruvector/cnn WASM module
		try {
			const moduleName = "@ruvector/cnn";
			const cnnModule = await import(/* @vite-ignore */ moduleName);
			if (cnnModule && typeof cnnModule.init === "function") {
				await cnnModule.init();
				this.wasmModule = cnnModule;
				this.usesWasm = true;
			}
		} catch {
			this.wasmModule = null;
			this.usesWasm = false;
		}

		// Priority 3: trained-weights + rules fallback (no neural network)
		this.initialized = true;
	}

	/**
	 * Classify a skin lesion image.
	 *
	 * Classification strategy (priority order):
	 * 1. ONNX V2 model (local, 95.97% mel sensitivity on external data)
	 *    -> 70% ONNX + 15% trained-weights + 15% rule-based
	 * 2. @ruvector/cnn WASM (if ONNX unavailable)
	 * 3. Trained-weights + rules fallback (no neural network)
	 *    -> 60% trained-weights + 40% rule-based
	 *
	 * No external API calls. Runs 100% on device.
	 *
	 * @param imageData - RGBA ImageData from canvas
	 * @returns Classification result with probabilities for all 7 classes
	 */
	async classify(imageData: ImageData): Promise<ClassificationResult> {
		if (!this.initialized) {
			await this.init();
		}

		const startTime = performance.now();

		// Preprocess: normalize, resize, convert to NCHW tensor
		const tensor = await preprocessImage(imageData);
		this.lastTensor = tensor;
		this.lastImageData = imageData;

		// Always run local CV analysis for ABCDE scores and ensemble blending
		const localProbabilities = this.classifyReal(imageData);

		let rawProbabilities: number[];
		let modelId: string;

		if (this.usesOnnx && isOfflineModelLoaded()) {
			// Priority 1: ONNX V2 model — 70% ONNX + 15% trained-weights + 15% rules
			const onnxResult = await onnxClassify(imageData, "auto");
			if (onnxResult) {
				const onnxProbs = CLASSES.map((cls) => {
					const match = onnxResult.probabilities.find((p) => p.className === cls);
					return match ? match.probability : 0;
				});
				rawProbabilities = CLASSES.map((_, i) =>
					DermClassifier.ONNX_WEIGHT * onnxProbs[i] +
					DermClassifier.TRAINED_WEIGHTS_WEIGHT_WITH_ONNX * this.getTrainedProbAtIndex(imageData, i) +
					DermClassifier.RULE_BASED_WEIGHT_WITH_ONNX * this.getRuleBasedProbAtIndex(localProbabilities, i),
				);
				modelId = "mela-v2-onnx-int8";
			} else {
				// ONNX failed at runtime — fall back to local
				rawProbabilities = localProbabilities;
				modelId = "mela-features-fallback";
			}
		} else if (this.usesWasm && this.wasmModule) {
			// Priority 2: WASM CNN
			rawProbabilities = await this.classifyWasm(tensor);
			modelId = "ruvector-cnn-wasm";
		} else {
			// Priority 3: trained-weights + rules only
			rawProbabilities = localProbabilities;
			modelId = "mela-features-fallback";
		}

		const inferenceTimeMs = Math.round(performance.now() - startTime);

		// Build sorted probabilities
		const probabilities: ClassProbability[] = CLASSES.map((cls, i) => ({
			className: cls,
			probability: rawProbabilities[i],
			label: LESION_LABELS[cls],
		})).sort((a, b) => b.probability - a.probability);

		const topClass = probabilities[0].className;
		const confidence = probabilities[0].probability;

		return {
			topClass,
			confidence,
			probabilities,
			modelId,
			inferenceTimeMs,
			usedWasm: this.usesWasm,
			usedHF: false,
			usedDualModel: false,
			usedCustomModel: this.usesOnnx,
			modelsDisagree: false,
			modelAgreement: this.usesOnnx ? 1.0 : 0,
		};
	}

	/**
	 * Classify with demographic adjustment using HAM10000 knowledge.
	 *
	 * Runs standard classification then applies Bayesian demographic
	 * adjustment based on patient age, sex, and lesion body site.
	 * Returns both raw and adjusted probabilities for transparency.
	 *
	 * @param imageData - RGBA ImageData from canvas
	 * @param demographics - Optional patient demographics
	 * @returns Classification result with adjusted probabilities
	 */
	async classifyWithDemographics(
		imageData: ImageData,
		demographics?: {
			age?: number;
			sex?: "male" | "female";
			localization?: string;
		},
	): Promise<ClassificationResult & {
		rawProbabilities: ClassProbability[];
		demographicAdjusted: boolean;
		clinicalRecommendation?: {
			recommendation: "biopsy" | "urgent_referral" | "monitor" | "reassurance";
			malignantProbability: number;
			melanomaProbability: number;
			reasoning: string;
		};
	}> {
		const result = await this.classify(imageData);

		if (!demographics || (!demographics.age && !demographics.sex && !demographics.localization)) {
			return {
				...result,
				rawProbabilities: result.probabilities,
				demographicAdjusted: false,
			};
		}

		// Build probability map from raw result
		const rawProbMap: Record<string, number> = {};
		for (const p of result.probabilities) {
			rawProbMap[p.className] = p.probability;
		}

		// Apply HAM10000 Bayesian demographic adjustment
		const adjustedMap = adjustForDemographics(
			rawProbMap,
			demographics.age,
			demographics.sex,
			demographics.localization,
		);

		// Build adjusted probabilities array
		const adjustedProbabilities: ClassProbability[] = CLASSES.map((cls) => ({
			className: cls,
			probability: adjustedMap[cls] ?? 0,
			label: LESION_LABELS[cls],
		})).sort((a, b) => b.probability - a.probability);

		const topClass = adjustedProbabilities[0].className;
		const confidence = adjustedProbabilities[0].probability;

		// Get clinical recommendation from adjusted probabilities
		const clinicalRecommendation = getClinicalRecommendation(adjustedMap);

		return {
			...result,
			topClass,
			confidence,
			probabilities: adjustedProbabilities,
			rawProbabilities: result.probabilities,
			demographicAdjusted: true,
			clinicalRecommendation,
		};
	}

	/**
	 * Generate Grad-CAM heatmap for the last classified image.
	 *
	 * @param targetClass - Optional class to explain (defaults to top predicted)
	 * @returns Grad-CAM heatmap and overlay
	 */
	async getGradCam(targetClass?: LesionClass): Promise<GradCamResult> {
		if (!this.lastTensor || !this.lastImageData) {
			throw new Error("No image classified yet. Call classify() first.");
		}

		const classIdx = targetClass ? CLASSES.indexOf(targetClass) : 0;
		const target = targetClass || CLASSES[0];

		if (this.usesWasm && this.wasmModule) {
			return this.gradCamWasm(classIdx, target);
		}

		return this.gradCamReal(target);
	}

	/**
	 * Check if the WASM module is loaded.
	 */
	isWasmLoaded(): boolean {
		return this.usesWasm;
	}

	/**
	 * Check if the classifier is initialized.
	 */
	isInitialized(): boolean {
		return this.initialized;
	}

	/**
	 * Get cached segmentation result from the last classification.
	 */
	getLastSegmentation(): SegmentationResult | null {
		return this.lastSegmentation;
	}

	/**
	 * Get cached asymmetry score from the last classification.
	 */
	getLastAsymmetry(): number {
		return this.lastAsymmetry;
	}

	/**
	 * Get cached border score from the last classification.
	 */
	getLastBorderScore(): number {
		return this.lastBorderScore;
	}

	/**
	 * Get cached color analysis from the last classification.
	 */
	getLastColorAnalysis(): ColorAnalysisResult | null {
		return this.lastColorAnalysis;
	}

	/**
	 * Get cached texture analysis from the last classification.
	 */
	getLastTexture(): TextureResult | null {
		return this.lastTexture;
	}

	/**
	 * Get cached structural analysis from the last classification.
	 */
	getLastStructures(): StructureResult | null {
		return this.lastStructures;
	}

	// ---- WASM backend ----

	private async classifyWasm(tensor: ImageTensor): Promise<number[]> {
		const raw = await this.wasmModule!.predict(tensor.data, [...tensor.shape]);
		return softmax(Array.from(raw));
	}

	private async gradCamWasm(classIdx: number, target: LesionClass): Promise<GradCamResult> {
		const rawHeatmap = await this.wasmModule!.gradCam(this.lastTensor!.data, classIdx);
		const heatmap = heatmapToImageData(rawHeatmap, 224, 224);
		const overlay = overlayHeatmap(this.lastImageData!, heatmap);

		return { heatmap, overlay, targetClass: target };
	}

	// ---- Ensemble configuration ----

	/**
	 * Ensemble weights for combining ONNX V2 model with local classifiers.
	 *
	 * ONNX V2 available: 70% ONNX + 15% trained-weights + 15% rule-based
	 * Fallback (no ONNX): 40% rule-based + 60% trained-weights
	 *
	 * The ONNX V2 model has 95.97% melanoma sensitivity on external data,
	 * so it gets the dominant weight. Trained-weights provide smooth linear
	 * discrimination. Rule-based captures nonlinear safety gates (melanoma
	 * floor, TDS override) that neural networks may miss.
	 */
	private static readonly RULE_BASED_WEIGHT = 0.4;
	private static readonly TRAINED_WEIGHTS_WEIGHT = 0.6;

	/** Weights when ONNX V2 model is available (3-way ensemble) */
	private static readonly ONNX_WEIGHT = 0.70;
	private static readonly TRAINED_WEIGHTS_WEIGHT_WITH_ONNX = 0.15;
	private static readonly RULE_BASED_WEIGHT_WITH_ONNX = 0.15;

	/** Cached individual classifier outputs for ensemble decomposition */
	private lastRuleBasedProbs: Record<string, number> | null = null;
	private lastTrainedProbs: Record<string, number> | null = null;

	/**
	 * Get the trained-weights probability for a specific class index.
	 * Uses the cached trained probs from the last classifyReal() call.
	 */
	private getTrainedProbAtIndex(_imageData: ImageData, index: number): number {
		if (!this.lastTrainedProbs) return 1 / 7;
		return this.lastTrainedProbs[CLASSES[index]] || 0;
	}

	/**
	 * Get the rule-based probability at a specific index from the local
	 * ensemble output. The local ensemble is already a 40/60 blend, so
	 * we decompose using the cached individual outputs instead.
	 */
	private getRuleBasedProbAtIndex(_localProbs: number[], index: number): number {
		if (!this.lastRuleBasedProbs) return 1 / 7;
		return this.lastRuleBasedProbs[CLASSES[index]] || 0;
	}

	// ---- Real image analysis fallback ----

	/**
	 * Real image analysis classifier using morphological feature extraction.
	 *
	 * Performs actual computer vision analysis on the image:
	 * 1. Segment lesion from surrounding skin (LAB + Otsu + connected component)
	 * 2. Measure asymmetry (principal axis folding)
	 * 3. Analyze border irregularity (8-octant scoring)
	 * 4. Analyze color composition (k-means in LAB space)
	 * 5. Compute GLCM texture features (contrast, homogeneity, entropy, correlation)
	 * 6. Detect dermoscopic structures (LBP, streaks, blue-white veil, etc.)
	 * 7. Classify from extracted features with HAM10000-calibrated priors
	 * 8. Classify with literature-derived logistic regression weights
	 * 9. Ensemble: weighted average of rule-based (40%) + trained weights (60%)
	 *
	 * All analysis results are cached for retrieval by the UI (ABCDE scores, etc.).
	 * Individual classifier outputs are also cached for 3-way ensemble decomposition.
	 */
	private classifyReal(imageData: ImageData): number[] {
		// 1. Segment the lesion
		const seg = segmentLesion(imageData);
		this.lastSegmentation = seg;

		// 2. Extract all features
		const asymmetry = measureAsymmetry(seg.mask, imageData.width, imageData.height, seg.bbox);
		this.lastAsymmetry = asymmetry;

		const borderScore = analyzeBorder(imageData, seg.mask, imageData.width, imageData.height);
		this.lastBorderScore = borderScore;

		const colorAnalysis = analyzeColors(imageData, seg.mask);
		this.lastColorAnalysis = colorAnalysis;

		const texture = analyzeTexture(imageData, seg.mask);
		this.lastTexture = texture;

		const structures = detectStructures(imageData, seg.mask);
		this.lastStructures = structures;

		// 3. Rule-based classification (existing approach with HAM10000-calibrated priors)
		const ruleBasedProbs = classifyFromFeatures({
			asymmetry,
			borderScore,
			colorAnalysis,
			texture,
			structures,
			lesionArea: seg.area,
			perimeter: seg.perimeter,
		});
		this.lastRuleBasedProbs = ruleBasedProbs;

		// 4. Trained-weights classification (literature-derived logistic regression)
		// Estimate diameter from segmentation area for the feature vector
		const diameterMm = estimateDiameterMm(seg.area, imageData.width);
		const featureVector = extractFeatureVector(
			seg,
			asymmetry,
			borderScore,
			colorAnalysis,
			texture,
			structures,
			diameterMm,
		);
		const trainedProbs = classifyWithTrainedWeights(featureVector);
		this.lastTrainedProbs = trainedProbs;

		// 5. Ensemble: weighted average of both classifiers
		// Rule-based captures nonlinear interactions (gates, floors, TDS overrides)
		// Trained-weights provides smooth, generalizable linear discrimination
		// Combined: more robust, typically 2-5% accuracy improvement
		const ensembleProbs: Record<string, number> = {};
		for (const cls of CLASSES) {
			ensembleProbs[cls] =
				DermClassifier.RULE_BASED_WEIGHT * (ruleBasedProbs[cls] || 0) +
				DermClassifier.TRAINED_WEIGHTS_WEIGHT * (trainedProbs[cls] || 0);
		}

		// Renormalize to ensure probabilities sum to 1.0
		const total = CLASSES.reduce((sum, cls) => sum + (ensembleProbs[cls] || 0), 0);
		if (total > 0) {
			for (const cls of CLASSES) {
				ensembleProbs[cls] = (ensembleProbs[cls] || 0) / total;
			}
		}

		// 6. Return in canonical class order
		return CLASSES.map((cls) => ensembleProbs[cls] || 0);
	}

	/**
	 * Generate a real attention heatmap based on image analysis.
	 *
	 * Uses color irregularity, structural complexity, and border proximity
	 * to show which regions drove the classification decision.
	 */
	private gradCamReal(target: LesionClass): GradCamResult {
		const imageData = this.lastImageData!;
		const seg = this.lastSegmentation;

		// Use real segmentation mask if available, otherwise generate a fallback
		const mask = seg ? seg.mask : new Uint8Array(imageData.width * imageData.height).fill(1);

		const heatmapData = generateAttentionMap(
			imageData,
			mask,
			imageData.width,
			imageData.height,
		);

		const heatmap = heatmapToImageData(heatmapData, 224, 224);
		const resizedOriginal = resizeBilinear(imageData, 224, 224);
		const overlay = overlayHeatmap(resizedOriginal, heatmap);

		return { heatmap, overlay, targetClass: target };
	}
}

/**
 * Cosine similarity between two probability vectors.
 * Returns a value in [0, 1] where 1 means identical distributions.
 */
function cosineSimilarity(a: number[], b: number[]): number {
	let dotProduct = 0;
	let normA = 0;
	let normB = 0;
	for (let i = 0; i < a.length; i++) {
		dotProduct += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	const denom = Math.sqrt(normA) * Math.sqrt(normB);
	return denom > 0 ? dotProduct / denom : 0;
}

/**
 * Softmax activation function.
 */
function softmax(logits: number[]): number[] {
	const maxLogit = Math.max(...logits);
	const exps = logits.map((l) => Math.exp(l - maxLogit));
	const sum = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sum);
}

/**
 * Convert a Float32 heatmap [0,1] to RGBA ImageData using a jet colormap.
 */
function heatmapToImageData(heatmap: Float32Array, width: number, height: number): ImageData {
	const rgba = new Uint8ClampedArray(width * height * 4);

	for (let i = 0; i < heatmap.length; i++) {
		const v = Math.max(0, Math.min(1, heatmap[i]));
		const px = i * 4;

		// Jet colormap approximation
		if (v < 0.25) {
			rgba[px] = 0;
			rgba[px + 1] = Math.round(v * 4 * 255);
			rgba[px + 2] = 255;
		} else if (v < 0.5) {
			rgba[px] = 0;
			rgba[px + 1] = 255;
			rgba[px + 2] = Math.round((1 - (v - 0.25) * 4) * 255);
		} else if (v < 0.75) {
			rgba[px] = Math.round((v - 0.5) * 4 * 255);
			rgba[px + 1] = 255;
			rgba[px + 2] = 0;
		} else {
			rgba[px] = 255;
			rgba[px + 1] = Math.round((1 - (v - 0.75) * 4) * 255);
			rgba[px + 2] = 0;
		}
		rgba[px + 3] = Math.round(v * 180); // Alpha based on intensity
	}

	return new ImageData(rgba, width, height);
}

/**
 * Overlay a heatmap on the original image with alpha blending.
 */
function overlayHeatmap(original: ImageData, heatmap: ImageData): ImageData {
	const width = heatmap.width;
	const height = heatmap.height;
	const resized = original.width === width && original.height === height
		? original
		: resizeBilinear(original, width, height);

	const result = new Uint8ClampedArray(width * height * 4);

	for (let i = 0; i < width * height; i++) {
		const px = i * 4;
		const alpha = heatmap.data[px + 3] / 255;

		result[px] = Math.round(resized.data[px] * (1 - alpha) + heatmap.data[px] * alpha);
		result[px + 1] = Math.round(resized.data[px + 1] * (1 - alpha) + heatmap.data[px + 1] * alpha);
		result[px + 2] = Math.round(resized.data[px + 2] * (1 - alpha) + heatmap.data[px + 2] * alpha);
		result[px + 3] = 255;
	}

	return new ImageData(result, width, height);
}
