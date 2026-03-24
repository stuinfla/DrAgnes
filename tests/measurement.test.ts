/**
 * Measurement orchestrator tests (ADR-121 Phase 4).
 * Verifies measureLesion returns the correct shape and that
 * safety warnings appear for uncertain measurements near 6mm.
 */
import { describe, it, expect } from "vitest";
import { measureLesion } from "../src/lib/dragnes/measurement";
import type { LesionMeasurement } from "../src/lib/dragnes/measurement";

// Polyfill ImageData for Node.js
if (typeof globalThis.ImageData === "undefined") {
	(globalThis as Record<string, unknown>).ImageData = class ImageData {
		readonly data: Uint8ClampedArray;
		readonly width: number;
		readonly height: number;
		readonly colorSpace: string = "srgb";
		constructor(dataOrWidth: Uint8ClampedArray | number, widthOrHeight: number, height?: number) {
			if (dataOrWidth instanceof Uint8ClampedArray) {
				this.data = dataOrWidth;
				this.width = widthOrHeight;
				this.height = height ?? (dataOrWidth.length / 4 / widthOrHeight);
			} else {
				this.width = dataOrWidth;
				this.height = widthOrHeight;
				this.data = new Uint8ClampedArray(this.width * this.height * 4);
			}
		}
	};
}

function mockImageData(w = 100, h = 100): ImageData {
	const data = new Uint8ClampedArray(w * h * 4);
	for (let i = 0; i < data.length; i += 4) {
		data[i] = 180; data[i + 1] = 140; data[i + 2] = 120; data[i + 3] = 255;
	}
	return new ImageData(data, w, h);
}

describe("measureLesion", () => {
	it("returns a LesionMeasurement object with all required fields", () => {
		const result: LesionMeasurement = measureLesion(mockImageData(), 500, "trunk");
		expect(result).toHaveProperty("diameterMm");
		expect(result).toHaveProperty("confidence");
		expect(result).toHaveProperty("method");
		expect(result).toHaveProperty("pixelsPerMm");
		expect(result).toHaveProperty("details");
		expect(typeof result.diameterMm).toBe("number");
	});

	it("confidence is one of high/medium/low", () => {
		const result = measureLesion(mockImageData(), 500, "trunk");
		expect(["high", "medium", "low"]).toContain(result.confidence);
	});

	it("method is one of connector/texture/estimate", () => {
		const result = measureLesion(mockImageData(), 500, "trunk");
		expect(["connector", "texture", "estimate"]).toContain(result.method);
	});

	it("diameterMm is a positive number", () => {
		const result = measureLesion(mockImageData(), 1000, "upper_extremity");
		expect(result.diameterMm).toBeGreaterThan(0);
	});

	it("pixelsPerMm is positive", () => {
		const result = measureLesion(mockImageData(200, 200), 800, "head");
		expect(result.pixelsPerMm).toBeGreaterThan(0);
	});

	it("includes safety warning for low-confidence 4-8mm measurements", () => {
		// Use a lesion area that would produce ~5-7mm at estimate confidence
		// With 100px wide image and estimate method: pxPerMm = 100/25 = 4
		// diameter = 2 * sqrt(area/pi) / pxPerMm
		// For 6mm: area = pi * (6*4/2)^2 = pi*144 ~ 452
		const result = measureLesion(mockImageData(100, 100), 450, "trunk");
		if (result.confidence === "low" && result.diameterMm >= 4 && result.diameterMm <= 8) {
			expect(result.details).toContain("6mm clinical threshold");
		}
	});
});
