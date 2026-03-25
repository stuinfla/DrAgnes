/**
 * CV Pipeline barrel export.
 *
 * Re-exports every public symbol from the decomposed CV modules
 * so that consumers can import from a single path.
 */

// Types
export type {
	BBox,
	SegmentationResult,
	ColorAnalysisResult,
	TextureResult,
	StructureResult,
	LesionPresenceResult,
} from "./types";

// Color space helpers
export { rgbToLab, toGrayscale } from "./color-space";

// Morphological operations
export {
	otsuThreshold,
	largestConnectedComponent,
	morphDilate,
	morphErode,
	morphClose,
	morphOpen,
} from "./morphology";

// Lesion segmentation
export { segmentLesion } from "./segmentation";

// Asymmetry measurement
export { measureAsymmetry } from "./asymmetry";

// Border analysis
export { analyzeBorder } from "./border";

// Color analysis
export { analyzeColors, kMeansLab } from "./color-analysis";

// Texture analysis (GLCM)
export { analyzeTexture } from "./texture-glcm";

// Structural pattern detection
export { detectStructures } from "./structures";

// Attention heatmap
export { generateAttentionMap, gaussianSmooth, resizeFloat32 } from "./attention-map";

// Feature-based classification
export { classifyFromFeatures } from "./feature-classifier";

// Measurement utilities
export { estimateDiameterMm, computeRiskLevel, detectLesionPresence } from "./measurement-utils";
