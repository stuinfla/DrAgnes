/**
 * DrAgnes - Dermoscopy CNN Classification Pipeline
 *
 * Browser-based skin lesion classification using MobileNetV3 WASM
 * with ABCDE dermoscopic scoring and privacy-preserving analytics.
 *
 * @module dragnes
 */

// Core classifier
export { DermClassifier } from "./classifier";

// ABCDE scoring
export { computeABCDE } from "./abcde";

// Image analysis engine
export {
	segmentLesion as segmentLesionReal,
	measureAsymmetry,
	analyzeBorder,
	analyzeColors,
	analyzeTexture,
	detectStructures,
	classifyFromFeatures,
	generateAttentionMap,
	estimateDiameterMm,
	computeRiskLevel,
	detectLesionPresence,
} from "./image-analysis";
export type { LesionPresenceResult } from "./image-analysis";

// Trained-weights classifier (literature-derived logistic regression)
export {
	classifyWithTrainedWeights,
	extractFeatureVector,
	TRAINED_WEIGHTS,
	FEATURE_NAMES,
	CLASS_NAMES,
} from "./trained-weights";
export type { ClassWeights } from "./trained-weights";

// HuggingFace ViT classifier (server-proxied, HAM10000 fine-tuned)
export {
	classifyWithHF,
	mapHFResultsToClasses,
	imageDataToBlob,
	LABEL_MAP as HF_LABEL_MAP,
	HF_MODEL,
	HF_API_URL,
} from "./hf-classifier";
export type { HFClassificationResult } from "./hf-classifier";

// Preprocessing pipeline
export {
	preprocessImage,
	colorNormalize,
	removeHair,
	segmentLesion,
	resizeBilinear,
	toNCHWTensor,
} from "./preprocessing";

// Privacy pipeline
export { PrivacyPipeline } from "./privacy";

// Configuration
export { DRAGNES_CONFIG } from "./config";
export type { DrAgnesConfig } from "./config";

// Consumer-friendly translation
export { translateForConsumer } from "./consumer-translation";
export type { ConsumerResult, ConsumerRiskLevel } from "./consumer-translation";

// ICD-10 code mapping
export { getPrimaryICD10, getLocationSpecificICD10, ICD10_MAP } from "./icd10";
export type { ICD10Code } from "./icd10";

// Multi-image consensus classification
export { classifyMultiImage, scoreImageQuality } from "./multi-image";
export type { MultiImageResult, ImageQualityScore } from "./multi-image";

// Image quality gating (ADR-121)
export { assessImageQuality } from "./image-quality";
export type { ImageQualityResult, QualityCheck } from "./image-quality";

// Types
export type {
	ABCDEScores,
	BodyLocation,
	ClassificationResult,
	ClassProbability,
	DermImage,
	DiagnosisRecord,
	GradCamResult,
	ImageTensor,
	LesionClass,
	LesionClassification,
	PatientEmbedding,
	PrivacyReport,
	RiskLevel,
	SegmentationMask,
	WitnessChain,
} from "./types";

export { LESION_LABELS } from "./types";
