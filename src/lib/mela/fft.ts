/**
 * Pure TypeScript 2D FFT (Cooley-Tukey radix-2). ADR-121 Phase 3.
 * All input sizes must be powers of 2.
 */

/** Smallest power of 2 >= n. */
export function nextPow2(n: number): number {
	let p = 1;
	while (p < n) p <<= 1;
	return p;
}

/** Zero-pad a 1D array to length `len`. */
export function zeroPad(arr: Float64Array, len: number): Float64Array {
	if (arr.length === len) return arr;
	const out = new Float64Array(len);
	out.set(arr);
	return out;
}

/** In-place Cooley-Tukey radix-2 FFT. Arrays must share a power-of-2 length. */
export function fft1d(real: Float64Array, imag: Float64Array): void {
	const n = real.length;
	for (let i = 1, j = 0; i < n; i++) {
		let bit = n >> 1;
		while (j & bit) { j ^= bit; bit >>= 1; }
		j ^= bit;
		if (i < j) {
			let t = real[i]; real[i] = real[j]; real[j] = t;
			t = imag[i]; imag[i] = imag[j]; imag[j] = t;
		}
	}
	for (let size = 2; size <= n; size <<= 1) {
		const half = size >> 1;
		const ang = -2 * Math.PI / size;
		const wR = Math.cos(ang), wI = Math.sin(ang);
		for (let i = 0; i < n; i += size) {
			let cR = 1, cI = 0;
			for (let j = 0; j < half; j++) {
				const eR = real[i + j], eI = imag[i + j];
				const oR = real[i + j + half] * cR - imag[i + j + half] * cI;
				const oI = real[i + j + half] * cI + imag[i + j + half] * cR;
				real[i + j] = eR + oR; imag[i + j] = eI + oI;
				real[i + j + half] = eR - oR; imag[i + j + half] = eI - oI;
				const tR = cR * wR - cI * wI; cI = cR * wI + cI * wR; cR = tR;
			}
		}
	}
}

/** 2D FFT via row-then-column 1D FFTs. Returns complex { real, imag }. */
export function fft2d(
	data: Float64Array, width: number, height: number,
): { real: Float64Array; imag: Float64Array } {
	const real = new Float64Array(data);
	const imag = new Float64Array(width * height);
	const rowR = new Float64Array(width), rowI = new Float64Array(width);
	for (let y = 0; y < height; y++) {
		const off = y * width;
		rowR.set(real.subarray(off, off + width)); rowI.fill(0);
		fft1d(rowR, rowI);
		real.set(rowR, off); imag.set(rowI, off);
	}
	const colR = new Float64Array(height), colI = new Float64Array(height);
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) { colR[y] = real[y * width + x]; colI[y] = imag[y * width + x]; }
		fft1d(colR, colI);
		for (let y = 0; y < height; y++) { real[y * width + x] = colR[y]; imag[y * width + x] = colI[y]; }
	}
	return { real, imag };
}

/** Power spectrum (magnitude squared) of a 2D real signal. */
export function powerSpectrum2d(data: Float64Array, width: number, height: number): Float64Array {
	const { real, imag } = fft2d(data, width, height);
	const ps = new Float64Array(real.length);
	for (let i = 0; i < ps.length; i++) ps[i] = real[i] * real[i] + imag[i] * imag[i];
	return ps;
}
