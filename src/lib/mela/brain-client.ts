/**
 * Mela Brain Integration Client
 *
 * Connects to the pi.ruv.io collective intelligence brain for:
 *   - Sharing de-identified lesion classifications
 *   - Searching similar cases
 *   - Enriching diagnoses with PubMed literature
 *   - Syncing LoRA model updates
 *
 * All data is stripped of PHI and has differential privacy noise applied
 * before leaving the device.
 */

import type { LesionClass, BodyLocation, WitnessChain } from "./types";
import type { AnonymizedCase } from "./anonymization";
import { createWitnessChain } from "./witness";
import { OfflineQueue } from "./offline-queue";

const BRAIN_BASE_URL = "https://pi.ruv.io";
const MELA_TAG = "mela";
const DEFAULT_EPSILON = 1.0;
const FETCH_TIMEOUT_MS = 10_000;

/** Metadata accompanying a brain contribution */
export interface DiagnosisMetadata {
	/** Predicted lesion class */
	lesionClass: LesionClass;
	/** Body location of the lesion */
	bodyLocation: BodyLocation;
	/** Model version that produced the classification */
	modelVersion: string;
	/** Confidence score [0, 1] */
	confidence: number;
	/** Per-class probabilities */
	probabilities: number[];
	/** Whether a clinician confirmed the diagnosis */
	confirmed: boolean;
	/** Brain epoch at time of classification */
	brainEpoch?: number;
}

/** A similar case returned from brain search */
export interface SimilarCase {
	/** Brain memory ID */
	id: string;
	/** Similarity score [0, 1] */
	similarity: number;
	/** Lesion class of the similar case */
	lesionClass: string;
	/** Body location */
	bodyLocation: string;
	/** Confidence of the original classification */
	confidence: number;
	/** Whether it was clinician-confirmed */
	confirmed: boolean;
}

/** Literature reference from brain + PubMed context */
export interface LiteratureReference {
	/** Title of the reference */
	title: string;
	/** Source (e.g. "PubMed", "brain-collective") */
	source: string;
	/** Summary or abstract excerpt */
	summary: string;
	/** URL if available */
	url?: string;
}

/** Mela-specific brain statistics */
export interface MelaStats {
	/** Total number of cases in the collective */
	totalCases: number;
	/** Cases per lesion class */
	casesByClass: Record<string, number>;
	/** Brain health status */
	brainStatus: string;
	/** Current brain epoch */
	epoch: number;
}

/** Result of sharing a diagnosis */
export interface ShareResult {
	/** Whether the share succeeded (or was queued offline) */
	success: boolean;
	/** Brain memory ID if online, null if queued */
	memoryId: string | null;
	/** Witness chain for the classification */
	witnessChain: WitnessChain[];
	/** Whether the contribution was queued for later sync */
	queued: boolean;
}

// ---- Differential Privacy ----

/**
 * Sample from a Laplace distribution with location 0 and scale b.
 */
function laplaceSample(scale: number): number {
	const u = Math.random() - 0.5;
	return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}

/**
 * Apply Laplace differential privacy noise to an embedding vector.
 *
 * @param embedding - Original embedding
 * @param epsilon - Privacy budget (lower = more noise)
 * @param sensitivity - L1 sensitivity of the embedding (default 1.0)
 * @returns New array with DP noise added
 */
function addDPNoise(embedding: number[], epsilon: number, sensitivity = 1.0): number[] {
	const scale = sensitivity / epsilon;
	return embedding.map((v) => v + laplaceSample(scale));
}

/**
 * Strip any potential PHI from metadata before sending to brain.
 * Only allows known safe fields through.
 */
function stripPHI(metadata: DiagnosisMetadata): Record<string, unknown> {
	return {
		lesionClass: metadata.lesionClass,
		bodyLocation: metadata.bodyLocation,
		modelVersion: metadata.modelVersion,
		confidence: metadata.confidence,
		confirmed: metadata.confirmed,
	};
}

// ---- Fetch helper ----

/**
 * Fetch with timeout. Throws on network error or timeout.
 */
async function fetchWithTimeout(
	url: string,
	options: RequestInit = {},
	timeoutMs = FETCH_TIMEOUT_MS
): Promise<Response> {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);

	try {
		const response = await fetch(url, {
			...options,
			signal: controller.signal,
		});
		return response;
	} finally {
		clearTimeout(timer);
	}
}

// ---- Brain Client ----

/** Singleton offline queue instance */
let offlineQueue: OfflineQueue | null = null;

function getOfflineQueue(): OfflineQueue {
	if (!offlineQueue) {
		offlineQueue = new OfflineQueue(BRAIN_BASE_URL);
	}
	return offlineQueue;
}

/**
 * Share a de-identified diagnosis with the pi.ruv.io brain.
 *
 * Pipeline:
 *   1. Strip all PHI from metadata
 *   2. Apply Laplace differential privacy noise (epsilon=1.0)
 *   3. Create witness chain hash
 *   4. POST to brain with mela tags
 *   5. If offline, queue for later sync
 *
 * @param embedding - Raw embedding vector (will have DP noise added)
 * @param metadata - Classification metadata (will have PHI stripped)
 * @returns ShareResult with witness chain and memory ID
 */
export async function shareDiagnosis(
	embedding: number[],
	metadata: DiagnosisMetadata
): Promise<ShareResult> {
	// Step 1: Strip PHI
	const safeMetadata = stripPHI(metadata);

	// Step 2: Apply differential privacy noise
	const dpEmbedding = addDPNoise(embedding, DEFAULT_EPSILON);

	// Step 3: Create witness chain
	const witnessChain = await createWitnessChain({
		embedding: dpEmbedding,
		modelVersion: metadata.modelVersion,
		probabilities: metadata.probabilities,
		brainEpoch: metadata.brainEpoch ?? 0,
		finalResult: metadata.lesionClass,
		confidence: metadata.confidence,
	});

	const witnessHash = witnessChain[witnessChain.length - 1].hash;

	// Step 4: Build brain memory payload
	const category = metadata.confirmed ? "solution" : "pattern";
	const tags = [
		MELA_TAG,
		`class:${metadata.lesionClass}`,
		`location:${metadata.bodyLocation}`,
		category,
	];

	const payload = {
		title: `Mela ${metadata.lesionClass} classification`,
		content: JSON.stringify({
			...safeMetadata,
			witnessHash,
			epsilon: DEFAULT_EPSILON,
		}),
		tags,
		category,
		embedding: dpEmbedding,
	};

	// Step 5: Attempt to send, queue if offline
	try {
		const response = await fetchWithTimeout(`${BRAIN_BASE_URL}/v1/memories`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(payload),
		});

		if (response.ok) {
			const result = (await response.json()) as { id?: string };
			return {
				success: true,
				memoryId: result.id ?? null,
				witnessChain,
				queued: false,
			};
		}

		// Non-OK response: queue for retry
		await getOfflineQueue().enqueue("/v1/memories", payload);
		return { success: true, memoryId: null, witnessChain, queued: true };
	} catch {
		// Network error: queue for later
		await getOfflineQueue().enqueue("/v1/memories", payload);
		return { success: true, memoryId: null, witnessChain, queued: true };
	}
}

/**
 * Search the brain for similar lesion embeddings.
 *
 * @param embedding - Query embedding (DP noise is added before search)
 * @param k - Number of results to return (default 5)
 * @returns Array of similar cases from the collective
 */
export async function searchSimilar(embedding: number[], k = 5): Promise<SimilarCase[]> {
	const dpEmbedding = addDPNoise(embedding, DEFAULT_EPSILON);

	try {
		const params = new URLSearchParams({
			q: JSON.stringify(dpEmbedding.slice(0, 16)),
			limit: String(k),
			tag: MELA_TAG,
		});

		const response = await fetchWithTimeout(`${BRAIN_BASE_URL}/v1/search?${params}`);

		if (!response.ok) {
			return [];
		}

		const data = (await response.json()) as {
			results?: Array<{
				id: string;
				similarity?: number;
				content?: string;
				tags?: string[];
			}>;
		};

		if (!data.results) {
			return [];
		}

		return data.results.map((r) => {
			let parsed: Record<string, unknown> = {};
			try {
				parsed = JSON.parse(r.content ?? "{}") as Record<string, unknown>;
			} catch {
				// content might not be JSON
			}

			return {
				id: r.id,
				similarity: r.similarity ?? 0,
				lesionClass: (parsed.lesionClass as string) ?? "unknown",
				bodyLocation: (parsed.bodyLocation as string) ?? "unknown",
				confidence: (parsed.confidence as number) ?? 0,
				confirmed: (parsed.confirmed as boolean) ?? false,
			};
		});
	} catch {
		return [];
	}
}

/**
 * Search brain and trigger PubMed context for literature references.
 *
 * @param lesionClass - The lesion class to search literature for
 * @returns Array of literature references
 */
export async function searchLiterature(lesionClass: LesionClass): Promise<LiteratureReference[]> {
	try {
		const params = new URLSearchParams({
			q: `${lesionClass} dermoscopy diagnosis treatment`,
			tag: MELA_TAG,
		});

		const response = await fetchWithTimeout(`${BRAIN_BASE_URL}/v1/search?${params}`);

		if (!response.ok) {
			return [];
		}

		const data = (await response.json()) as {
			results?: Array<{
				title?: string;
				content?: string;
				tags?: string[];
				url?: string;
			}>;
		};

		if (!data.results) {
			return [];
		}

		return data.results.map((r) => ({
			title: r.title ?? "Untitled",
			source: r.tags?.includes("pubmed") ? "PubMed" : "brain-collective",
			summary: (r.content ?? "").slice(0, 500),
			url: r.url,
		}));
	} catch {
		return [];
	}
}

/**
 * Check for LoRA model updates from the collective brain.
 *
 * @returns Object with update availability and version info, or null if offline
 */
export async function syncModel(): Promise<{
	available: boolean;
	version: string | null;
	epoch: number;
} | null> {
	try {
		const response = await fetchWithTimeout(`${BRAIN_BASE_URL}/v1/status`);

		if (!response.ok) {
			return null;
		}

		const status = (await response.json()) as {
			epoch?: number;
			version?: string;
			loraAvailable?: boolean;
		};

		return {
			available: status.loraAvailable ?? false,
			version: status.version ?? null,
			epoch: status.epoch ?? 0,
		};
	} catch {
		return null;
	}
}

/**
 * Get Mela-specific brain statistics.
 *
 * @returns Statistics about the collective, or null if offline
 */
export async function getStats(): Promise<MelaStats | null> {
	try {
		const [statusRes, searchRes] = await Promise.all([
			fetchWithTimeout(`${BRAIN_BASE_URL}/v1/status`),
			fetchWithTimeout(
				`${BRAIN_BASE_URL}/v1/search?${new URLSearchParams({ q: "*", tag: MELA_TAG, limit: "0" })}`
			),
		]);

		if (!statusRes.ok) {
			return null;
		}

		const status = (await statusRes.json()) as {
			status?: string;
			epoch?: number;
			totalMemories?: number;
		};

		let totalCases = status.totalMemories ?? 0;
		const casesByClass: Record<string, number> = {};

		if (searchRes.ok) {
			const searchData = (await searchRes.json()) as {
				total?: number;
				results?: Array<{ content?: string }>;
			};
			totalCases = searchData.total ?? totalCases;

			if (searchData.results) {
				for (const r of searchData.results) {
					try {
						const parsed = JSON.parse(r.content ?? "{}") as { lesionClass?: string };
						if (parsed.lesionClass) {
							casesByClass[parsed.lesionClass] =
								(casesByClass[parsed.lesionClass] ?? 0) + 1;
						}
					} catch {
						// skip unparseable entries
					}
				}
			}
		}

		return {
			totalCases,
			casesByClass,
			brainStatus: status.status ?? "unknown",
			epoch: status.epoch ?? 0,
		};
	} catch {
		return null;
	}
}

/**
 * Get the offline queue instance for manual queue management.
 */
export function getQueue(): OfflineQueue {
	return getOfflineQueue();
}

// ---- ADR-126 Phase 2: Anonymized Case Sharing ----

const BRAIN_API = `${BRAIN_BASE_URL}/v1`;

/**
 * Share an anonymized case with the pi-brain collective intelligence.
 *
 * Posts the de-identified case (probabilities with DP noise, demographics
 * reduced to decade/sex, no images or identifiers) under the
 * "dermatology_case" category with mela tags.
 *
 * Fails gracefully -- brain connectivity is optional and should never
 * block classification workflow.
 *
 * @param case_ - AnonymizedCase produced by the anonymization pipeline
 * @param apiKey - Optional API key for authenticated brain access
 * @returns true if the share succeeded, false on any failure
 */
export async function shareToBrain(
	case_: AnonymizedCase,
	apiKey?: string
): Promise<boolean> {
	try {
		const headers: Record<string, string> = {
			"Content-Type": "application/json",
		};
		if (apiKey) {
			headers["Authorization"] = `Bearer ${apiKey}`;
		}

		const payload = {
			title: `Mela ${case_.topClass} classification`,
			content: JSON.stringify(case_),
			category: "dermatology_case",
			tags: [
				MELA_TAG,
				`class:${case_.topClass}`,
				`location:${case_.bodyLocation}`,
				case_.outcome ?? "no_outcome",
			],
		};

		const response = await fetchWithTimeout(`${BRAIN_API}/memories/share`, {
			method: "POST",
			headers,
			body: JSON.stringify(payload),
		});

		return response.ok;
	} catch {
		return false;
	}
}

/**
 * Search pi-brain for cases with similar probability distributions.
 *
 * Serializes the probability vector into a query string and searches
 * the dermatology_case category for nearest neighbors. Returns empty
 * array on any failure -- brain is never a hard dependency.
 *
 * @param probabilities - Probability map (class name to noised probability)
 * @param limit - Maximum number of similar cases to return (default 5)
 * @param apiKey - Optional API key for authenticated brain access
 * @returns Array of similar AnonymizedCase records, or empty on failure
 */
export async function searchSimilarCases(
	probabilities: Record<string, number>,
	limit: number = 5,
	apiKey?: string
): Promise<AnonymizedCase[]> {
	try {
		const headers: Record<string, string> = {};
		if (apiKey) {
			headers["Authorization"] = `Bearer ${apiKey}`;
		}

		// Build a compact query from the top classes for semantic search
		const sorted = Object.entries(probabilities)
			.sort(([, a], [, b]) => b - a)
			.slice(0, 3);
		const query = sorted
			.map(([cls, prob]) => `${cls}:${prob.toFixed(3)}`)
			.join(" ");

		const params = new URLSearchParams({
			q: `dermatology_case ${query}`,
			limit: String(limit),
			tag: MELA_TAG,
		});

		const response = await fetchWithTimeout(
			`${BRAIN_API}/memories/search?${params}`,
			{ headers }
		);

		if (!response.ok) {
			return [];
		}

		const data = (await response.json()) as {
			results?: Array<{
				content?: string;
			}>;
		};

		if (!data.results) {
			return [];
		}

		const cases: AnonymizedCase[] = [];
		for (const r of data.results) {
			try {
				const parsed = JSON.parse(r.content ?? "{}") as AnonymizedCase;
				// Validate that it has the minimum required fields
				if (parsed.topClass && parsed.probabilities) {
					cases.push(parsed);
				}
			} catch {
				// Skip entries that are not valid AnonymizedCase JSON
			}
		}

		return cases;
	} catch {
		return [];
	}
}

/**
 * Check pi-brain health status.
 *
 * Lightweight GET to the status endpoint. Returns a summary of
 * brain health including total memories, graph edges, and whether
 * the service is responsive.
 *
 * @returns Status object, or a degraded response with healthy=false on failure
 */
export async function getBrainStatus(): Promise<{
	memories: number;
	edges: number;
	healthy: boolean;
}> {
	try {
		const response = await fetchWithTimeout(`${BRAIN_API}/status`);

		if (!response.ok) {
			return { memories: 0, edges: 0, healthy: false };
		}

		const data = (await response.json()) as {
			totalMemories?: number;
			memories?: number;
			edges?: number;
			totalEdges?: number;
			status?: string;
		};

		return {
			memories: data.totalMemories ?? data.memories ?? 0,
			edges: data.totalEdges ?? data.edges ?? 0,
			healthy: data.status === "ok" || data.status === "healthy",
		};
	} catch {
		return { memories: 0, edges: 0, healthy: false };
	}
}
