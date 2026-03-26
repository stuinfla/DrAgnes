/**
 * Type stub for onnxruntime-node.
 *
 * onnxruntime-node is externalized from the Vite build (see vite.config.ts)
 * and only used server-side in the classify-local endpoint. This stub
 * provides enough type information to satisfy the checker without
 * requiring the native binary to be installed at development time.
 */
declare module 'onnxruntime-node' {
	export const InferenceSession: {
		create(path: string, options?: unknown): Promise<{
			run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>>;
			inputNames: string[];
			outputNames: string[];
		}>;
	};
	export class Tensor {
		constructor(type: string, data: Float32Array | Float64Array | Int32Array | BigInt64Array, dims: number[]);
		readonly data: Float32Array;
		readonly dims: number[];
		readonly type: string;
	}
}
