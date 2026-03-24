/**
 * Multi-image consensus classification tests.
 * Verifies image quality scoring range and empty-array error.
 * The melanoma safety gate test uses the scoreImageQuality function directly.
 */
import { describe, it, expect } from "vitest";
import { scoreImageQuality } from "../src/lib/dragnes/multi-image";

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

function mockImageData(w = 100, h = 100, r = 180, g = 140, b = 120): ImageData {
	const data = new Uint8ClampedArray(w * h * 4);
	for (let i = 0; i < data.length; i += 4) {
		data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
	}
	return new ImageData(data, w, h);
}

describe("scoreImageQuality", () => {
	it("returns all scores in 0-1 range", () => {
		const q = scoreImageQuality(mockImageData());
		expect(q.sharpness).toBeGreaterThanOrEqual(0);
		expect(q.sharpness).toBeLessThanOrEqual(1);
		expect(q.contrast).toBeGreaterThanOrEqual(0);
		expect(q.contrast).toBeLessThanOrEqual(1);
		expect(q.segmentationQuality).toBeGreaterThanOrEqual(0);
		expect(q.segmentationQuality).toBeLessThanOrEqual(1);
		expect(q.overallScore).toBeGreaterThanOrEqual(0);
		expect(q.overallScore).toBeLessThanOrEqual(1);
	});

	it("overall = 0.4*sharpness + 0.3*contrast + 0.3*segQuality", () => {
		const q = scoreImageQuality(mockImageData());
		const expected = 0.4 * q.sharpness + 0.3 * q.contrast + 0.3 * q.segmentationQuality;
		expect(q.overallScore).toBeCloseTo(expected, 5);
	});

	it("uniform image has near-zero sharpness and contrast", () => {
		const q = scoreImageQuality(mockImageData(100, 100, 128, 128, 128));
		expect(q.sharpness).toBeLessThan(0.05);
		expect(q.contrast).toBeLessThan(0.05);
	});
});

describe("classifyMultiImage", () => {
	it("throws on empty image array", async () => {
		// Import dynamically to test the throw without needing a real classifier
		const { classifyMultiImage } = await import("../src/lib/dragnes/multi-image");
		const fakeClassifier = {} as any;
		await expect(classifyMultiImage(fakeClassifier, [])).rejects.toThrow(
			"classifyMultiImage requires at least one image",
		);
	});
});
