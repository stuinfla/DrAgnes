/**
 * Shared type definitions for the CV pipeline modules.
 */

export interface BBox {
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface SegmentationResult {
	mask: Uint8Array;
	bbox: BBox;
	area: number;
	perimeter: number;
}

export interface ColorAnalysisResult {
	colorCount: number;
	dominantColors: Array<{ name: string; percentage: number; rgb: [number, number, number] }>;
	hasBlueWhiteStructures: boolean;
}

export interface TextureResult {
	contrast: number;
	homogeneity: number;
	entropy: number;
	correlation: number;
}

export interface StructureResult {
	hasIrregularNetwork: boolean;
	hasIrregularGlobules: boolean;
	hasStreaks: boolean;
	hasBlueWhiteVeil: boolean;
	hasRegressionStructures: boolean;
	structuralScore: number;
}

export interface LesionPresenceResult {
	hasLesion: boolean;
	confidence: number;
	reason: string;
}
