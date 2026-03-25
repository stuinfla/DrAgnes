import { describe, it, expect } from "vitest";
import { detectSpots, type SpotDetection } from "../src/lib/mela/spot-detector";

// Polyfill ImageData for Node.js
if (typeof globalThis.ImageData === "undefined") {
	(globalThis as any).ImageData = class ImageData {
		data: Uint8ClampedArray;
		width: number;
		height: number;
		constructor(data: Uint8ClampedArray, width: number, height?: number) {
			this.data = data;
			this.width = width;
			this.height = height ?? (data.length / 4 / width);
		}
	};
}

function makeImageData(
	w: number,
	h: number,
	fill: (x: number, y: number) => [number, number, number, number],
): ImageData {
	const data = new Uint8ClampedArray(w * h * 4);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			const [r, g, b, a] = fill(x, y);
			const i = (y * w + x) * 4;
			data[i] = r;
			data[i + 1] = g;
			data[i + 2] = b;
			data[i + 3] = a;
		}
	}
	return new ImageData(data, w, h);
}

describe("detectSpots", () => {
	it("rejects uniform skin (no spots) — plain hand", () => {
		// Uniform pinkish-brown skin with slight texture noise
		const img = makeImageData(200, 200, (x, y) => {
			const noise = Math.sin(x * 0.1 + y * 0.1) * 3;
			return [195 + noise, 155 + noise, 135 + noise, 255];
		});
		const result = detectSpots(img);
		expect(result.hasSpot).toBe(false);
		expect(result.spotCount).toBe(0);
		expect(result.confidence).toBeGreaterThan(0.5);
	});

	it("detects a dark circular spot on skin — mole", () => {
		// Light skin background with a round dark brown mole in center
		const img = makeImageData(200, 200, (x, y) => {
			const cx = 100, cy = 100, r = 18;
			const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
			if (dist < r) {
				// Dark brown mole (~L*30)
				return [55, 28, 18, 255];
			}
			// Normal skin (~L*70)
			return [200, 160, 140, 255];
		});
		const result = detectSpots(img);
		expect(result.hasSpot).toBe(true);
		expect(result.spotCount).toBeGreaterThanOrEqual(1);
		expect(result.largestSpotArea).toBeGreaterThan(0);
	});

	it("rejects a shadow gradient — not compact, no blob", () => {
		// Skin with a broad linear shadow on the right side
		const img = makeImageData(200, 200, (x, _y) => {
			// Gradual darkening from left to right — NOT a spot
			const factor = 1 - (x / 200) * 0.5;
			return [
				Math.round(200 * factor),
				Math.round(160 * factor),
				Math.round(140 * factor),
				255,
			];
		});
		const result = detectSpots(img);
		// The gradient creates a large, non-compact dark region.
		// Pass 1 may fire but Pass 2 should reject (compactness too low or area too large).
		expect(result.hasSpot).toBe(false);
	});

	it("detects a small 3mm mole on a back-sized image", () => {
		// Simulates a small mole (~3mm) on someone's back.
		// At 200x200 representing ~10cm field of view, 3mm = ~6px diameter.
		const img = makeImageData(200, 200, (x, y) => {
			const cx = 120, cy = 80, r = 4; // ~3mm mole, off-center
			const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
			if (dist < r) {
				return [50, 25, 15, 255]; // very dark mole
			}
			return [195, 155, 135, 255]; // uniform skin
		});
		const result = detectSpots(img);
		// A 4px radius circle has area ~50 pixels. On 200x200 (40000 pixels)
		// that is 0.125% — above the 0.1% minimum threshold.
		expect(result.hasSpot).toBe(true);
		expect(result.spotCount).toBeGreaterThanOrEqual(1);
	});

	it("detects multiple moles — spotCount > 1", () => {
		// Two distinct dark moles on skin
		const img = makeImageData(200, 200, (x, y) => {
			const d1 = Math.sqrt((x - 60) ** 2 + (y - 60) ** 2);
			const d2 = Math.sqrt((x - 140) ** 2 + (y - 140) ** 2);
			if (d1 < 12) return [50, 25, 15, 255]; // mole 1
			if (d2 < 10) return [55, 30, 20, 255]; // mole 2
			return [200, 160, 140, 255]; // skin
		});
		const result = detectSpots(img);
		expect(result.hasSpot).toBe(true);
		expect(result.spotCount).toBeGreaterThanOrEqual(2);
	});

	it("rejects completely uniform gray image", () => {
		const img = makeImageData(200, 200, () => [128, 128, 128, 255]);
		const result = detectSpots(img);
		expect(result.hasSpot).toBe(false);
	});

	it("returns hasSpot: true for an off-center lesion", () => {
		// Mole in top-left corner — tests that we scan entire image
		const img = makeImageData(200, 200, (x, y) => {
			const dist = Math.sqrt((x - 30) ** 2 + (y - 30) ** 2);
			if (dist < 15) return [45, 22, 12, 255];
			return [200, 160, 140, 255];
		});
		const result = detectSpots(img);
		expect(result.hasSpot).toBe(true);
	});

	it("returns confidence and reason fields", () => {
		const img = makeImageData(200, 200, () => [200, 160, 140, 255]);
		const result = detectSpots(img);
		expect(typeof result.confidence).toBe("number");
		expect(result.confidence).toBeGreaterThanOrEqual(0);
		expect(result.confidence).toBeLessThanOrEqual(1);
		expect(typeof result.reason).toBe("string");
		expect(result.reason.length).toBeGreaterThan(0);
	});
});
