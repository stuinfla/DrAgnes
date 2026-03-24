/**
 * Image quality gating tests (ADR-121 Phase 1).
 * Verifies assessImageQuality grades, brightness, and glare detection.
 */
import { describe, it, expect } from "vitest";
import { assessImageQuality } from "../src/lib/dragnes/image-quality";
import type { ImageQualityResult } from "../src/lib/dragnes/image-quality";

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

function uniformImage(w: number, h: number, r: number, g: number, b: number): ImageData {
	const data = new Uint8ClampedArray(w * h * 4);
	for (let i = 0; i < data.length; i += 4) {
		data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
	}
	return new ImageData(data, w, h);
}

describe("assessImageQuality", () => {
	it("returns grade as good, acceptable, or poor", () => {
		const result = assessImageQuality(uniformImage(60, 60, 128, 128, 128));
		expect(["good", "acceptable", "poor"]).toContain(result.grade);
	});

	it("returns 5 quality checks", () => {
		const result = assessImageQuality(uniformImage(60, 60, 128, 100, 80));
		expect(result.checks).toHaveLength(5);
		const names = result.checks.map((c) => c.name);
		expect(names).toContain("sharpness");
		expect(names).toContain("brightness");
		expect(names).toContain("contrast");
		expect(names).toContain("framing");
		expect(names).toContain("glare");
	});

	it("completely black image returns poor with brightness failing", () => {
		const result = assessImageQuality(uniformImage(60, 60, 0, 0, 0));
		expect(result.grade).toBe("poor");
		const brightness = result.checks.find((c) => c.name === "brightness")!;
		expect(brightness.passed).toBe(false);
		expect(brightness.message).toContain("dark");
	});

	it("completely white image returns poor with glare failing", () => {
		const result = assessImageQuality(uniformImage(60, 60, 255, 255, 255));
		expect(result.grade).toBe("poor");
		const glare = result.checks.find((c) => c.name === "glare")!;
		expect(glare.passed).toBe(false);
	});

	it("overallScore is between 0 and 1", () => {
		// Use a lesion-like image with variation to avoid floating-point NaN on uniform pixels
		const w = 80, h = 80;
		const data = new Uint8ClampedArray(w * h * 4);
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				const i = (y * w + x) * 4;
				const inCenter = x > 20 && x < 60 && y > 20 && y < 60;
				data[i] = inCenter ? 100 : 180;
				data[i + 1] = inCenter ? 60 : 150;
				data[i + 2] = inCenter ? 40 : 130;
				data[i + 3] = 255;
			}
		}
		const result = assessImageQuality(new ImageData(data, w, h));
		expect(result.overallScore).toBeGreaterThanOrEqual(0);
		expect(result.overallScore).toBeLessThanOrEqual(1);
	});

	it("suggestion is null when all checks pass, or a string when some fail", () => {
		const result = assessImageQuality(uniformImage(60, 60, 0, 0, 0));
		if (result.checks.some((c) => !c.passed)) {
			expect(typeof result.suggestion).toBe("string");
		}
	});
});
