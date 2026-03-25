/**
 * Color space conversion helpers.
 *
 * Single source of truth for RGB-to-LAB and grayscale conversions
 * used throughout the CV pipeline.
 */

export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
	// Normalize to [0,1]
	let rn = r / 255;
	let gn = g / 255;
	let bn = b / 255;

	// sRGB to linear
	rn = rn > 0.04045 ? Math.pow((rn + 0.055) / 1.055, 2.4) : rn / 12.92;
	gn = gn > 0.04045 ? Math.pow((gn + 0.055) / 1.055, 2.4) : gn / 12.92;
	bn = bn > 0.04045 ? Math.pow((bn + 0.055) / 1.055, 2.4) : bn / 12.92;

	// Linear RGB to XYZ (D65 illuminant)
	let x = (rn * 0.4124564 + gn * 0.3575761 + bn * 0.1804375) / 0.95047;
	let y = (rn * 0.2126729 + gn * 0.7151522 + bn * 0.0721750);
	let z = (rn * 0.0193339 + gn * 0.1191920 + bn * 0.9503041) / 1.08883;

	// XYZ to LAB
	const epsilon = 0.008856;
	const kappa = 903.3;
	x = x > epsilon ? Math.cbrt(x) : (kappa * x + 16) / 116;
	y = y > epsilon ? Math.cbrt(y) : (kappa * y + 16) / 116;
	z = z > epsilon ? Math.cbrt(z) : (kappa * z + 16) / 116;

	const L = 116 * y - 16;
	const a = 500 * (x - y);
	const bLab = 200 * (y - z);

	return [L, a, bLab];
}

export function toGrayscale(imageData: ImageData): Uint8Array {
	const { data, width, height } = imageData;
	const gray = new Uint8Array(width * height);
	for (let i = 0; i < gray.length; i++) {
		const px = i * 4;
		gray[i] = Math.round(0.299 * data[px] + 0.587 * data[px + 1] + 0.114 * data[px + 2]);
	}
	return gray;
}
