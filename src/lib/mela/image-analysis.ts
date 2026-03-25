/**
 * Mela Dermoscopic Image Analysis Engine
 *
 * This file is now a thin re-export layer. All implementation has been
 * decomposed into focused modules under ./cv/. Existing imports from
 * "$lib/mela/image-analysis" continue to work unchanged.
 *
 * Modules:
 *   cv/types.ts          - BBox, SegmentationResult, ColorAnalysisResult, etc.
 *   cv/color-space.ts    - rgbToLab, toGrayscale
 *   cv/morphology.ts     - otsuThreshold, morphDilate/Erode/Close/Open, largestConnectedComponent
 *   cv/segmentation.ts   - segmentLesion
 *   cv/asymmetry.ts      - measureAsymmetry
 *   cv/border.ts         - analyzeBorder
 *   cv/color-analysis.ts - analyzeColors, kMeansLab
 *   cv/texture-glcm.ts   - analyzeTexture
 *   cv/structures.ts     - detectStructures
 *   cv/attention-map.ts  - generateAttentionMap, gaussianSmooth, resizeFloat32
 *   cv/feature-classifier.ts - classifyFromFeatures
 *   cv/measurement-utils.ts  - estimateDiameterMm, computeRiskLevel, detectLesionPresence
 */

// Re-export everything from the CV pipeline barrel
export {
	// Types
	type BBox,
	type SegmentationResult,
	type ColorAnalysisResult,
	type TextureResult,
	type StructureResult,
	type LesionPresenceResult,

	// Color space helpers
	rgbToLab,
	toGrayscale,

	// Morphological operations
	otsuThreshold,
	largestConnectedComponent,
	morphDilate,
	morphErode,
	morphClose,
	morphOpen,

	// Lesion segmentation
	segmentLesion,

	// Asymmetry measurement
	measureAsymmetry,

	// Border analysis
	analyzeBorder,

	// Color analysis
	analyzeColors,
	kMeansLab,

	// Texture analysis (GLCM)
	analyzeTexture,

	// Structural pattern detection
	detectStructures,

	// Attention heatmap
	generateAttentionMap,
	gaussianSmooth,
	resizeFloat32,

	// Feature-based classification
	classifyFromFeatures,

	// Measurement utilities
	estimateDiameterMm,
	computeRiskLevel,
	detectLesionPresence,
} from "./cv/index";
