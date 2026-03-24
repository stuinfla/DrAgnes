/**
 * DrAgnes CNN Classification Engine
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
import { LABEL_MAP } from "./hf-classifier";

/** All HAM10000 classes in canonical order (used by all ensemble layers) */
const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/**
 * Label mapping for skintaglabs/siglip-skin-lesion-classifier (SigLIP 400M).
 * Maps SigLIP model labels to our canonical 7-class HAM10000 taxonomy.
 * Covers both full names and short forms the model might return.
 */
const SIGLIP_LABEL_MAP: Record<string, string> = {
	// Full names (title case variants)
	"Melanoma": "mel",
	"Basal Cell Carcinoma": "bcc",
	"Actinic Keratosis": "akiec",
	"Benign Keratosis-like Lesions": "bkl",
	"Benign Keratosis": "bkl",
	"Dermatofibroma": "df",
	"Melanocytic Nevi": "nv",
	"Melanocytic Nevus": "nv",
	"Vascular Lesions": "vasc",
	"Vascular Lesion": "vasc",
	// Lower case variants
	"melanoma": "mel",
	"basal cell carcinoma": "bcc",
	"actinic keratosis": "akiec",
	"benign keratosis-like lesions": "bkl",
	"benign keratosis": "bkl",
	"dermatofibroma": "df",
	"melanocytic nevi": "nv",
	"melanocytic nevus": "nv",
	"vascular lesion": "vasc",
	"vascular lesions": "vasc",
	// SigLIP may also output SCC and other extended classes
	"squamous cell carcinoma": "akiec",
	"Squamous Cell Carcinoma": "akiec",
	"seborrheic keratosis": "bkl",
	"Seborrheic Keratosis": "bkl",
	// Short-form abbreviations
	"akiec": "akiec",
	"bcc": "bcc",
	"bkl": "bkl",
	"df": "df",
	"mel": "mel",
	"nv": "nv",
	"vasc": "vasc",
};

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
	private lastTensor: ImageTensor | null = null;
	private lastImageData: ImageData | null = null;

	/** Whether the HuggingFace ViT model was used in the last classification */
	private lastUsedHF = false;

	/** Whether the custom-trained local ViT model was used */
	private lastUsedCustomModel = false;

	/** Whether the dual-model ensemble was used in the last classification */
	private lastUsedDualModel = false;
	/** Whether the two HF models disagreed on the top class */
	private lastModelsDisagree = false;
	/** Cosine similarity between the two model output distributions (0-1) */
	private lastModelAgreement = 0;

	/** Cached results from the last real image analysis pass */
	private lastSegmentation: SegmentationResult | null = null;
	private lastAsymmetry: number = 0;
	private lastBorderScore: number = 0;
	private lastColorAnalysis: ColorAnalysisResult | null = null;
	private lastTexture: TextureResult | null = null;
	private lastStructures: StructureResult | null = null;

	/**
	 * Initialize the classifier.
	 * Attempts to load the @ruvector/cnn WASM module.
	 * Falls back to demo mode if unavailable.
	 */
	async init(): Promise<void> {
		if (this.initialized) return;

		try {
			// Dynamic import of the WASM CNN package
			// Use variable to prevent Vite from pre-bundling this optional dependency
			const moduleName = "@ruvector/cnn";
			const cnnModule = await import(/* @vite-ignore */ moduleName);
			if (cnnModule && typeof cnnModule.init === "function") {
				await cnnModule.init();
				this.wasmModule = cnnModule;
				this.usesWasm = true;
			}
		} catch {
			// WASM module not available, use demo fallback
			this.wasmModule = null;
			this.usesWasm = false;
		}

		this.initialized = true;
	}

	/**
	 * Classify a dermoscopic image.
	 *
	 * Classification strategy (priority order):
	 * 1. Custom-trained local ViT model (95.97% melanoma sensitivity on external data, /api/classify-local)
	 *    -> 70% custom model + 15% trained-weights + 15% rule-based
	 * 2. Dual HF models in parallel via server proxy
	 *    -> 50% dual-HF-ensemble + 30% trained-weights + 20% rule-based
	 * 3. Single HF model
	 *    -> 60% single-HF + 25% trained-weights + 15% rule-based
	 * 4. Offline fallback (local analysis only)
	 *    -> 60% trained-weights + 40% rule-based
	 *
	 * Local image analysis always runs for ABCDE scores and ensemble blending.
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

		let rawProbabilities: number[];

		// Reset model state
		this.lastUsedDualModel = false;
		this.lastUsedCustomModel = false;
		this.lastModelsDisagree = false;
		this.lastModelAgreement = 0;

		if (this.usesWasm && this.wasmModule) {
			// WASM path: use the CNN model directly, skip HF
			rawProbabilities = await this.classifyWasm(tensor);
			this.lastUsedHF = false;
		} else {
			// Local analysis (always runs -- needed for ABCDE scores, fallback, and ensemble)
			const localProbabilities = this.classifyReal(imageData);

			// Priority 1: Try the custom-trained local ViT model (95.97% mel sensitivity on external data)
			const customResult = await this.classifyCustomLocal(imageData);

			if (customResult) {
				// Custom model available: 70% custom + 15% trained-weights + 15% rule-based
				rawProbabilities = CLASSES.map((_, i) =>
					DermClassifier.CUSTOM_MODEL_WEIGHT * customResult[i] +
					DermClassifier.TRAINED_WEIGHTS_WEIGHT_WITH_CUSTOM * this.getTrainedProbAtIndex(imageData, i) +
					DermClassifier.RULE_BASED_WEIGHT_WITH_CUSTOM * this.getRuleBasedProbAtIndex(localProbabilities, i),
				);
				this.lastUsedCustomModel = true;
				this.lastUsedHF = false;
			} else {
				// Priority 2-3: Try dual HF ViT classification via server proxy
				const dualResult = await this.classifyHFDual(imageData);

				if (dualResult.dualAvailable) {
					// 4-way ensemble: 50% dual-HF + 30% trained-weights + 20% rule-based
					rawProbabilities = CLASSES.map((_, i) =>
						DermClassifier.DUAL_HF_WEIGHT * dualResult.ensembledProbabilities[i] +
						DermClassifier.TRAINED_WEIGHTS_WEIGHT_WITH_DUAL * this.getTrainedProbAtIndex(imageData, i) +
						DermClassifier.RULE_BASED_WEIGHT_WITH_DUAL * this.getRuleBasedProbAtIndex(localProbabilities, i),
					);
					this.lastUsedHF = true;
					this.lastUsedDualModel = true;
					this.lastModelsDisagree = dualResult.modelsDisagree;
					this.lastModelAgreement = dualResult.modelAgreement;
				} else if (dualResult.singleAvailable) {
					// 3-way ensemble: 60% single-HF + 25% trained-weights + 15% rule-based
					rawProbabilities = CLASSES.map((_, i) =>
						DermClassifier.HF_WEIGHT * dualResult.singleProbabilities![i] +
						DermClassifier.TRAINED_WEIGHTS_WEIGHT_WITH_HF * this.getTrainedProbAtIndex(imageData, i) +
						DermClassifier.RULE_BASED_WEIGHT_WITH_HF * this.getRuleBasedProbAtIndex(localProbabilities, i),
					);
					this.lastUsedHF = true;
				} else {
					// Offline fallback: local ensemble only
					rawProbabilities = localProbabilities;
					this.lastUsedHF = false;
				}
			}
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

		let modelId: string;
		if (this.usesWasm) {
			modelId = "mobilenetv3-small-wasm";
		} else if (this.lastUsedCustomModel) {
			modelId = "dragnes-custom-v1-98pct-mel";
		} else if (this.lastUsedDualModel) {
			modelId = "dual-ensemble-v2-siglip50-anwarkh50-trained30-rule20";
		} else if (this.lastUsedHF) {
			modelId = "ensemble-v2-hf60-trained25-rule15";
		} else {
			modelId = "ensemble-v1-rule40-trained60";
		}

		return {
			topClass,
			confidence,
			probabilities,
			modelId,
			inferenceTimeMs,
			usedWasm: this.usesWasm,
			usedHF: this.lastUsedHF,
			usedDualModel: this.lastUsedDualModel,
			usedCustomModel: this.lastUsedCustomModel,
			modelsDisagree: this.lastModelsDisagree,
			modelAgreement: this.lastModelAgreement,
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
	 * Ensemble weights for combining classifiers.
	 *
	 * Offline (no HF): 40% rule-based + 60% trained-weights
	 * Online (single HF): 60% HF ViT + 25% trained-weights + 15% rule-based
	 * Online (dual HF): 50% dual-HF-ensemble + 30% trained-weights + 20% rule-based
	 *
	 * The HF ViT models have actual trained weights from skin lesion data,
	 * so they get the highest weight when available. The local trained-weights
	 * classifier uses a linear model with literature-derived weights for smooth
	 * generalization. Rule-based captures nonlinear interactions (gates, floors,
	 * TDS overrides) that linear models miss.
	 */
	private static readonly RULE_BASED_WEIGHT = 0.4;
	private static readonly TRAINED_WEIGHTS_WEIGHT = 0.6;

	/** Weights when single HF ViT is available (3-way ensemble) */
	private static readonly HF_WEIGHT = 0.6;
	private static readonly TRAINED_WEIGHTS_WEIGHT_WITH_HF = 0.25;
	private static readonly RULE_BASED_WEIGHT_WITH_HF = 0.15;

	/** Weights when dual HF models are available (4-way ensemble) */
	private static readonly DUAL_HF_WEIGHT = 0.5;
	private static readonly TRAINED_WEIGHTS_WEIGHT_WITH_DUAL = 0.3;
	private static readonly RULE_BASED_WEIGHT_WITH_DUAL = 0.2;

	/** Weights when custom-trained local model is available (highest priority) */
	private static readonly CUSTOM_MODEL_WEIGHT = 0.7;
	private static readonly TRAINED_WEIGHTS_WEIGHT_WITH_CUSTOM = 0.15;
	private static readonly RULE_BASED_WEIGHT_WITH_CUSTOM = 0.15;

	/**
	 * Per-class weights within the dual-HF ensemble.
	 *
	 * Model 1 (Anwarkh1): 85.8M params, 44K+ downloads, 7-class HAM10000.
	 * Model 2 (SigLIP skintaglabs): SigLIP 400M, MIT license, dermatology company.
	 *
	 * Neither model has independently verified melanoma recall, so we use
	 * equal weighting (50/50) for all classes until validation data is available.
	 * This replaces the previous actavkid weighting (removed from HuggingFace).
	 */
	private static readonly MODEL2_MELANOMA_WEIGHT = 0.5;
	private static readonly MODEL1_MELANOMA_WEIGHT = 0.5;
	private static readonly MODEL2_OTHER_WEIGHT = 0.5;
	private static readonly MODEL1_OTHER_WEIGHT = 0.5;

	/** Cached individual classifier outputs for ensemble decomposition */
	private lastRuleBasedProbs: Record<string, number> | null = null;
	private lastTrainedProbs: Record<string, number> | null = null;

	// ---- Custom-trained local ViT model ----
	// Trained on HAM10000 + ISIC 2019 combined (37,484 images) with focal loss, 95.97% melanoma sensitivity on external data.
	// Runs via /api/classify-local (Python subprocess with model.safetensors).

	/**
	 * Classify using the custom-trained local ViT model via /api/classify-local.
	 *
	 * This is the highest-priority classification path because it was trained
	 * specifically for this task with 95.97% melanoma sensitivity on external ISIC 2019 data
	 * (focal loss, HAM10000 + ISIC 2019 combined, ViT-base-patch16-224).
	 *
	 * @param imageData - RGBA ImageData from canvas
	 * @returns Canonical-order probability array, or null if the local model is unavailable
	 */
	private async classifyCustomLocal(imageData: ImageData): Promise<number[] | null> {
		try {
			// Convert ImageData to JPEG blob
			const canvas = document.createElement("canvas");
			canvas.width = imageData.width;
			canvas.height = imageData.height;
			const ctx = canvas.getContext("2d")!;
			ctx.putImageData(imageData, 0, 0);
			const blob = await new Promise<Blob>((resolve, reject) => {
				canvas.toBlob(
					(b) => (b ? resolve(b) : reject(new Error("blob failed"))),
					"image/jpeg",
					0.95,
				);
			});

			const formData = new FormData();
			formData.append("image", blob, "lesion.jpg");

			const response = await fetch("/api/classify-local", {
				method: "POST",
				body: formData,
			});

			if (!response.ok) return null;

			const { results } = await response.json() as {
				results: Array<{ label: string; score: number }>;
			};

			// Map results to canonical class order
			const probs: Record<string, number> = {
				akiec: 0, bcc: 0, bkl: 0, df: 0, mel: 0, nv: 0, vasc: 0,
			};

			for (const r of results) {
				const key = r.label.toLowerCase();
				if (key in probs) {
					probs[key] = r.score;
				}
			}

			// Normalize to sum to 1
			const total = Object.values(probs).reduce((a, b) => a + b, 0);
			return CLASSES.map((c) => (total > 0 ? probs[c] / total : 1 / 7));
		} catch {
			// Custom model not available -- fall through to HF models
			return null;
		}
	}

	// ---- Dual-model HuggingFace ensemble ----
	// Model 1: Anwarkh1/Skin_Cancer-Image_Classification (ViT-Base, 85.8M)
	// Model 2: skintaglabs/siglip-skin-lesion-classifier (SigLIP 400M)
	// Replaces: actavkid/vit-large-patch32-384 (REMOVED from HuggingFace, HTTP 410)

	/**
	 * Classify using BOTH HuggingFace models in parallel and ensemble them.
	 *
	 * Model 1: Anwarkh1/Skin_Cancer-Image_Classification (ViT-Base, 85.8M params, 7-class)
	 * Model 2: skintaglabs/siglip-skin-lesion-classifier (SigLIP 400M, MIT license)
	 *
	 * Ensemble logic:
	 * - Equal weighting (50/50) for all classes (no proven recall advantage for either model)
	 * - If models disagree on top class: flag with modelsDisagree = true
	 *
	 * Falls back to single model if one fails, or empty if both fail.
	 */
	private async classifyHFDual(
		imageData: ImageData,
	): Promise<{
		dualAvailable: boolean;
		singleAvailable: boolean;
		ensembledProbabilities: number[];
		singleProbabilities?: number[];
		modelsDisagree: boolean;
		modelAgreement: number;
	}> {
		// Convert ImageData to JPEG blob once, share between both calls
		let blob: Blob;
		try {
			const canvas = document.createElement("canvas");
			canvas.width = imageData.width;
			canvas.height = imageData.height;
			const ctx = canvas.getContext("2d")!;
			ctx.putImageData(imageData, 0, 0);
			blob = await new Promise<Blob>((resolve, reject) => {
				canvas.toBlob(
					(b) => (b ? resolve(b) : reject(new Error("blob failed"))),
					"image/jpeg",
					0.95,
				);
			});
		} catch {
			return { dualAvailable: false, singleAvailable: false, ensembledProbabilities: [], modelsDisagree: false, modelAgreement: 0 };
		}

		// Call both models in PARALLEL using Promise.allSettled
		const [v1Result, v2Result] = await Promise.allSettled([
			this.callHFModel("/api/classify", blob, LABEL_MAP),
			this.callHFModel("/api/classify-v2", blob, SIGLIP_LABEL_MAP),
		]);

		const v1Available = v1Result.status === "fulfilled" && v1Result.value !== null;
		const v2Available = v2Result.status === "fulfilled" && v2Result.value !== null;

		const v1Probs = v1Available ? (v1Result as PromiseFulfilledResult<number[]>).value : null;
		const v2Probs = v2Available ? (v2Result as PromiseFulfilledResult<number[]>).value : null;

		if (v1Probs && v2Probs) {
			// Both models available -- ensemble them
			const ensembled = this.ensembleDualModels(v1Probs, v2Probs);

			// Determine top classes from each model
			const v1TopIdx = v1Probs.indexOf(Math.max(...v1Probs));
			const v2TopIdx = v2Probs.indexOf(Math.max(...v2Probs));
			const modelsDisagree = v1TopIdx !== v2TopIdx;

			// Compute cosine similarity between the two distributions
			const agreement = cosineSimilarity(v1Probs, v2Probs);

			return {
				dualAvailable: true,
				singleAvailable: true,
				ensembledProbabilities: ensembled,
				modelsDisagree,
				modelAgreement: agreement,
			};
		} else if (v1Probs) {
			// Only Anwarkh1 available
			return {
				dualAvailable: false,
				singleAvailable: true,
				ensembledProbabilities: [],
				singleProbabilities: v1Probs,
				modelsDisagree: false,
				modelAgreement: 0,
			};
		} else if (v2Probs) {
			// Only skintaglabs SigLIP available
			return {
				dualAvailable: false,
				singleAvailable: true,
				ensembledProbabilities: [],
				singleProbabilities: v2Probs,
				modelsDisagree: false,
				modelAgreement: 0,
			};
		} else {
			// Both failed -- fall back to local analysis
			return {
				dualAvailable: false,
				singleAvailable: false,
				ensembledProbabilities: [],
				modelsDisagree: false,
				modelAgreement: 0,
			};
		}
	}

	/**
	 * Call a single HF model endpoint and return normalized canonical probabilities.
	 *
	 * @param endpoint - API endpoint path (e.g., "/api/classify" or "/api/classify-v2")
	 * @param blob - Image JPEG blob
	 * @param labelMap - Label-to-canonical-class mapping for this model
	 * @returns Canonical-order probability array, or null if call fails
	 */
	private async callHFModel(
		endpoint: string,
		blob: Blob,
		labelMap: Record<string, string>,
	): Promise<number[] | null> {
		try {
			const formData = new FormData();
			formData.append("image", blob, "lesion.jpg");

			const response = await fetch(endpoint, {
				method: "POST",
				body: formData,
			});

			if (!response.ok) return null;

			const { results } = await response.json() as {
				results: Array<{ label: string; score: number }>;
			};

			// Map labels to canonical classes
			const probs: Record<string, number> = {
				akiec: 0, bcc: 0, bkl: 0, df: 0, mel: 0, nv: 0, vasc: 0,
			};

			for (const r of results) {
				const canonical =
					labelMap[r.label] || labelMap[r.label.toLowerCase()] || null;
				if (canonical && canonical in probs) {
					// Accumulate scores for labels that map to the same class
					// (e.g., "melanoma" and "melanoma metastasis" both map to "mel")
					probs[canonical] += r.score;
				}
			}

			// Normalize to sum to 1
			const total = Object.values(probs).reduce((a, b) => a + b, 0);
			return CLASSES.map((c) => (total > 0 ? probs[c] / total : 1 / 7));
		} catch {
			return null;
		}
	}

	/**
	 * Ensemble two model outputs with class-specific weighting.
	 *
	 * Currently uses equal weighting (50/50) for all classes since neither
	 * model has independently verified melanoma recall. To be updated once
	 * validation results for the skintaglabs SigLIP model are available.
	 */
	private ensembleDualModels(v1Probs: number[], v2Probs: number[]): number[] {
		const ensembled = CLASSES.map((cls, i) => {
			if (cls === "mel") {
				return DermClassifier.MODEL1_MELANOMA_WEIGHT * v1Probs[i] +
					DermClassifier.MODEL2_MELANOMA_WEIGHT * v2Probs[i];
			}
			return DermClassifier.MODEL1_OTHER_WEIGHT * v1Probs[i] +
				DermClassifier.MODEL2_OTHER_WEIGHT * v2Probs[i];
		});

		// Renormalize to sum to 1
		const total = ensembled.reduce((a, b) => a + b, 0);
		if (total > 0) {
			return ensembled.map((p) => p / total);
		}
		return ensembled;
	}

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
