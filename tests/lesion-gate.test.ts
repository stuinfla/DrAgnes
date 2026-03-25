import { describe, it, expect } from "vitest";
import { detectLesionPresence } from "../src/lib/mela/image-analysis";

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

function makeImageData(w: number, h: number, fill: (x: number, y: number) => [number, number, number, number]): ImageData {
	const data = new Uint8ClampedArray(w * h * 4);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			const [r, g, b, a] = fill(x, y);
			const i = (y * w + x) * 4;
			data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = a;
		}
	}
	return new ImageData(data, w, h);
}

describe("detectLesionPresence", () => {
	it("REJECTS uniform skin-colored image (no lesion)", () => {
		// Back of hand: uniform pinkish-brown, slight noise
		const img = makeImageData(200, 200, (x, y) => {
			const noise = Math.sin(x * 0.1 + y * 0.1) * 5;
			return [195 + noise, 155 + noise, 135 + noise, 255];
		});
		const result = detectLesionPresence(img);
		expect(result.hasLesion).toBe(false);
	});

	it("REJECTS uniform skin with subtle shadow (like between fingers)", () => {
		const img = makeImageData(200, 200, (x, y) => {
			// Skin base with a shadow gradient on the right side
			const shadow = x > 150 ? (x - 150) * 0.3 : 0;
			return [195 - shadow, 155 - shadow, 135 - shadow, 255];
		});
		const result = detectLesionPresence(img);
		expect(result.hasLesion).toBe(false);
	});

	it("ACCEPTS image with a clear dark lesion in center", () => {
		const img = makeImageData(200, 200, (x, y) => {
			const cx = 100, cy = 100, r = 25;
			const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
			if (dist < r) {
				// Dark brown lesion
				return [60, 30, 20, 255];
			}
			// Normal skin
			return [200, 160, 140, 255];
		});
		const result = detectLesionPresence(img);
		expect(result.hasLesion).toBe(true);
	});

	it("REJECTS completely uniform gray (not skin, no lesion)", () => {
		const img = makeImageData(200, 200, () => [128, 128, 128, 255]);
		const result = detectLesionPresence(img);
		expect(result.hasLesion).toBe(false);
	});

	it("ACCEPTS image with an asymmetric multi-colored lesion", () => {
		const img = makeImageData(200, 200, (x, y) => {
			const cx = 100, cy = 100;
			const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
			if (dist < 30) {
				// Multi-colored lesion: dark brown + black + reddish
				if (x < cx) return [40, 20, 15, 255];
				if (y < cy) return [80, 40, 30, 255];
				return [100, 20, 20, 255];
			}
			return [200, 160, 140, 255];
		});
		const result = detectLesionPresence(img);
		expect(result.hasLesion).toBe(true);
	});
});
