/**
 * Type stub for sharp.
 *
 * sharp is externalized from the Vite build (see vite.config.ts)
 * and only used server-side in the classify-local endpoint for image
 * preprocessing. This stub provides enough type information to satisfy
 * the checker without requiring the native binary at development time.
 */
declare module 'sharp' {
	interface SharpInstance {
		resize(width: number, height: number): SharpInstance;
		removeAlpha(): SharpInstance;
		raw(): SharpInstance;
		toBuffer(options: { resolveWithObject: true }): Promise<{ data: Buffer; info: { width: number; height: number; channels: number } }>;
		toBuffer(): Promise<Buffer>;
	}
	function sharp(input: Buffer | string): SharpInstance;
	export default sharp;
}
