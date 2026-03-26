/**
 * Practice Analytics Store -- tracks classification events and clinician
 * feedback; computes concordance, per-class accuracy, calibration, equity,
 * and trends.  Persists to localStorage.
 */
import type { LesionClass } from "$lib/mela/types";

export interface ClassificationEvent {
	id: string;
	timestamp: string;
	predictedClass: string;
	confidence: number;
	allProbabilities: Record<string, number>;
	modelId: string;
	demographics?: { age?: number; sex?: string; fitzpatrick?: number };
	bodyLocation?: string;
}

export interface OutcomeFeedback {
	eventId: string;
	timestamp: string;
	pathologyResult?: string;
	clinicianImpression?: string;
	concordant?: boolean;
	discordanceReason?: "overcalled" | "missed" | "artifact" | "edge_case" | "other";
	biopsied: boolean;
	malignant?: boolean;
}

export interface PerClassMetrics {
	className: string;
	sensitivity: number;
	specificity: number;
	ppv: number;
	npv: number;
	ci95: [number, number];
	n: number;
}

export interface CalibrationBin {
	binStart: number;
	binEnd: number;
	predictedMean: number;
	observedFrequency: number;
	count: number;
}

export interface FitzpatrickRow {
	type: number;
	accuracy: number;
	n: number;
}

export interface DiscordanceBreakdown {
	overcalled: number;
	missed: number;
	artifact: number;
	edge_case: number;
	other: number;
}

export interface DiscordantCase {
	eventId: string;
	timestamp: string;
	predictedClass: string;
	reason: string;
	pathologyResult?: string;
}

export interface RollingConcordance {
	date: string;
	concordanceRate: number;
	n: number;
}

export interface PracticeMetrics {
	concordanceRate: number;
	concordanceN: number;
	nnb: number;
	nnbN: number;
	totalScans: number;
	feedbackRate: number;
	feedbackN: number;
	perClass: PerClassMetrics[];
	ece: number;
	calibrationBins: CalibrationBin[];
	hosmerLemeshowP: number;
	fitzpatrick: FitzpatrickRow[];
	discordance: DiscordanceBreakdown;
	discordantCases: DiscordantCase[];
	rolling30Day: RollingConcordance[];
}

const STORAGE_KEY_EVENTS = "mela_analytics_events";
const STORAGE_KEY_FEEDBACK = "mela_analytics_feedback";

function loadFromStorage<T>(key: string): T[] {
	if (typeof window === "undefined") return [];
	try {
		const raw = localStorage.getItem(key);
		return raw ? JSON.parse(raw) : [];
	} catch {
		return [];
	}
}

/** Maximum total localStorage budget for analytics (4 MB). */
const MAX_STORAGE_BYTES = 4 * 1024 * 1024;

function saveToStorage<T>(key: string, data: T[]): void {
	if (typeof window === "undefined") return;
	try {
		const json = JSON.stringify(data);
		localStorage.setItem(key, json);

		// Trim oldest entries if total analytics storage exceeds 4 MB
		const eventsSize = localStorage.getItem(STORAGE_KEY_EVENTS)?.length ?? 0;
		const feedbackSize = localStorage.getItem(STORAGE_KEY_FEEDBACK)?.length ?? 0;
		const totalSize = (eventsSize + feedbackSize) * 2; // chars to approximate bytes (UTF-16)
		if (totalSize > MAX_STORAGE_BYTES && key === STORAGE_KEY_EVENTS && data.length > 10) {
			// Remove oldest 20% of events to free space
			const trimCount = Math.ceil(data.length * 0.2);
			const trimmed = data.slice(trimCount);
			localStorage.setItem(key, JSON.stringify(trimmed));
		}
	} catch {
		// localStorage full or unavailable -- silently fail
	}
}

let events: ClassificationEvent[] = [];
let feedback: OutcomeFeedback[] = [];
let loaded = false;

function ensureLoaded(): void {
	if (loaded) return;
	events = loadFromStorage<ClassificationEvent>(STORAGE_KEY_EVENTS);
	feedback = loadFromStorage<OutcomeFeedback>(STORAGE_KEY_FEEDBACK);
	loaded = true;
}

export function recordClassification(input: Omit<ClassificationEvent, "id" | "timestamp">): string {
	ensureLoaded();
	const id = crypto.randomUUID();
	const event: ClassificationEvent = {
		...input,
		id,
		timestamp: new Date().toISOString(),
	};
	events.push(event);
	saveToStorage(STORAGE_KEY_EVENTS, events);
	return id;
}

export function recordFeedback(input: Omit<OutcomeFeedback, "timestamp">): void {
	ensureLoaded();
	const entry: OutcomeFeedback = {
		...input,
		timestamp: new Date().toISOString(),
	};
	// Replace existing feedback for the same event if present
	const idx = feedback.findIndex((f) => f.eventId === input.eventId);
	if (idx >= 0) {
		feedback[idx] = entry;
	} else {
		feedback.push(entry);
	}
	saveToStorage(STORAGE_KEY_FEEDBACK, feedback);
}

export function getEvents(): ClassificationEvent[] {
	ensureLoaded();
	return events;
}

export function getFeedback(): OutcomeFeedback[] {
	ensureLoaded();
	return feedback;
}

export function getLastEventId(): string | null {
	ensureLoaded();
	return events.length > 0 ? events[events.length - 1].id : null;
}

export function clearAnalytics(): void {
	events = [];
	feedback = [];
	saveToStorage(STORAGE_KEY_EVENTS, events);
	saveToStorage(STORAGE_KEY_FEEDBACK, feedback);
}

/** Wilson score confidence interval for a binomial proportion. */
export function wilsonCI(successes: number, total: number, z: number = 1.96): [number, number] {
	if (total === 0) return [0, 0];
	const p = successes / total;
	const denom = 1 + (z * z) / total;
	const center = (p + (z * z) / (2 * total)) / denom;
	const margin = (z / denom) * Math.sqrt((p * (1 - p)) / total + (z * z) / (4 * total * total));
	return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

/** Chi-squared CDF approximation for Hosmer-Lemeshow test. */
function chiSquaredCDF(x: number, k: number): number {
	if (x <= 0) return 0;
	// Regularized lower incomplete gamma via series expansion
	const halfK = k / 2;
	const halfX = x / 2;
	let sum = 0;
	let term = Math.exp(-halfX) * Math.pow(halfX, halfK) / gamma(halfK + 1);
	for (let i = 0; i < 200; i++) {
		sum += term;
		term *= halfX / (halfK + 1 + i);
	}
	return Math.min(1, sum);
}

function gamma(z: number): number {
	if (z < 0.5) {
		return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
	}
	z -= 1;
	const g = 7;
	const c = [
		0.99999999999980993, 676.5203681218851, -1259.1392167224028,
		771.32342877765313, -176.61502916214059, 12.507343278686905,
		-0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
	];
	let x = c[0];
	for (let i = 1; i < g + 2; i++) {
		x += c[i] / (z + i);
	}
	const t = z + g + 0.5;
	return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

const ALL_CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/** Compute all practice metrics from events and feedback. */
export function computeMetrics(): PracticeMetrics {
	ensureLoaded();

	const feedbackMap = new Map<string, OutcomeFeedback>();
	for (const f of feedback) {
		feedbackMap.set(f.eventId, f);
	}

	const concordantFeedback = feedback.filter((f) => f.concordant !== undefined);
	const concordantCount = concordantFeedback.filter((f) => f.concordant === true).length;
	const concordanceRate = concordantFeedback.length > 0
		? concordantCount / concordantFeedback.length : 0;

	const biopsied = feedback.filter((f) => f.biopsied);
	const biopsiedMalignant = biopsied.filter((f) => f.malignant === true);
	const nnb = biopsiedMalignant.length > 0 ? biopsied.length / biopsiedMalignant.length : 0;
	const feedbackRate = events.length > 0 ? feedback.length / events.length : 0;

	const perClass = computePerClassMetrics(events, feedbackMap);
	const { ece, bins } = computeCalibration(events, feedbackMap);
	const hlP = computeHosmerLemeshow(bins);
	const fitzpatrick = computeFitzpatrickEquity(events, feedbackMap);
	const { breakdown, cases } = computeDiscordance(feedback, events);
	const rolling30Day = computeRolling30Day(events, feedbackMap);

	return {
		concordanceRate,
		concordanceN: concordantFeedback.length,
		nnb,
		nnbN: biopsied.length,
		totalScans: events.length,
		feedbackRate,
		feedbackN: feedback.length,
		perClass,
		ece,
		calibrationBins: bins,
		hosmerLemeshowP: hlP,
		fitzpatrick,
		discordance: breakdown,
		discordantCases: cases,
		rolling30Day,
	};
}

function computePerClassMetrics(
	evts: ClassificationEvent[],
	fbMap: Map<string, OutcomeFeedback>,
): PerClassMetrics[] {
	return ALL_CLASSES.map((cls) => {
		let tp = 0, fp = 0, tn = 0, fn = 0;

		for (const evt of evts) {
			const fb = fbMap.get(evt.id);
			if (!fb || !fb.pathologyResult) continue;

			const predicted = evt.predictedClass === cls;
			const actual = fb.pathologyResult === cls;

			if (predicted && actual) tp++;
			else if (predicted && !actual) fp++;
			else if (!predicted && actual) fn++;
			else tn++;
		}

		const total = tp + fp + tn + fn;
		const sensitivity = (tp + fn) > 0 ? tp / (tp + fn) : 0;
		const specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0;
		const ppv = (tp + fp) > 0 ? tp / (tp + fp) : 0;
		const npv = (tn + fn) > 0 ? tn / (tn + fn) : 0;
		const ci95 = wilsonCI(tp, tp + fn);

		return {
			className: cls,
			sensitivity,
			specificity,
			ppv,
			npv,
			ci95,
			n: tp + fn,
		};
	});
}

function computeCalibration(
	evts: ClassificationEvent[],
	fbMap: Map<string, OutcomeFeedback>,
): { ece: number; bins: CalibrationBin[] } {
	const NUM_BINS = 10;
	const bins: CalibrationBin[] = Array.from({ length: NUM_BINS }, (_, i) => ({
		binStart: i / NUM_BINS,
		binEnd: (i + 1) / NUM_BINS,
		predictedMean: 0,
		observedFrequency: 0,
		count: 0,
	}));

	// We calibrate on the predicted confidence vs whether the top prediction was correct
	let totalWithPathology = 0;

	for (const evt of evts) {
		const fb = fbMap.get(evt.id);
		if (!fb || !fb.pathologyResult) continue;

		totalWithPathology++;
		const conf = evt.confidence;
		const correct = evt.predictedClass === fb.pathologyResult ? 1 : 0;

		const binIdx = Math.min(NUM_BINS - 1, Math.floor(conf * NUM_BINS));
		bins[binIdx].predictedMean += conf;
		bins[binIdx].observedFrequency += correct;
		bins[binIdx].count++;
	}

	// Finalize bin means
	let ece = 0;
	for (const bin of bins) {
		if (bin.count > 0) {
			bin.predictedMean /= bin.count;
			bin.observedFrequency /= bin.count;
			ece += (bin.count / Math.max(1, totalWithPathology)) * Math.abs(bin.predictedMean - bin.observedFrequency);
		} else {
			bin.predictedMean = (bin.binStart + bin.binEnd) / 2;
			bin.observedFrequency = 0;
		}
	}

	return { ece, bins };
}

function computeHosmerLemeshow(bins: CalibrationBin[]): number {
	// H-L statistic: sum over bins of (O - E)^2 / (E * (1 - E/n))
	let hlStat = 0;
	const nonEmpty = bins.filter((b) => b.count > 0);

	for (const bin of nonEmpty) {
		const observed = bin.observedFrequency * bin.count;
		const expected = bin.predictedMean * bin.count;

		if (expected > 0 && expected < bin.count) {
			hlStat += ((observed - expected) ** 2) / (expected * (1 - expected / bin.count));
		}
	}

	// Degrees of freedom = number of groups - 2
	const df = Math.max(1, nonEmpty.length - 2);
	const cdf = chiSquaredCDF(hlStat, df);

	return Math.max(0, 1 - cdf);
}

function computeFitzpatrickEquity(
	evts: ClassificationEvent[],
	fbMap: Map<string, OutcomeFeedback>,
): FitzpatrickRow[] {
	const byType = new Map<number, { correct: number; total: number }>();

	for (const evt of evts) {
		const fst = evt.demographics?.fitzpatrick;
		if (fst === undefined || fst < 1 || fst > 6) continue;

		const fb = fbMap.get(evt.id);
		if (!fb || !fb.pathologyResult) continue;

		const entry = byType.get(fst) ?? { correct: 0, total: 0 };
		entry.total++;
		if (evt.predictedClass === fb.pathologyResult) entry.correct++;
		byType.set(fst, entry);
	}

	const rows: FitzpatrickRow[] = [];
	for (let t = 1; t <= 6; t++) {
		const entry = byType.get(t);
		rows.push({
			type: t,
			accuracy: entry && entry.total > 0 ? entry.correct / entry.total : 0,
			n: entry?.total ?? 0,
		});
	}

	return rows;
}

function computeDiscordance(
	allFeedback: OutcomeFeedback[],
	evts: ClassificationEvent[],
): { breakdown: DiscordanceBreakdown; cases: DiscordantCase[] } {
	const breakdown: DiscordanceBreakdown = {
		overcalled: 0,
		missed: 0,
		artifact: 0,
		edge_case: 0,
		other: 0,
	};

	const cases: DiscordantCase[] = [];
	const evtMap = new Map<string, ClassificationEvent>();
	for (const e of evts) evtMap.set(e.id, e);

	for (const fb of allFeedback) {
		if (fb.concordant !== false) continue;
		if (fb.discordanceReason && fb.discordanceReason in breakdown) {
			breakdown[fb.discordanceReason]++;
		} else {
			breakdown.other++;
		}

		const evt = evtMap.get(fb.eventId);
		cases.push({
			eventId: fb.eventId,
			timestamp: fb.timestamp,
			predictedClass: evt?.predictedClass ?? "unknown",
			reason: fb.discordanceReason ?? "other",
			pathologyResult: fb.pathologyResult,
		});
	}

	// Sort most recent first, limit to 20
	cases.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
	return { breakdown, cases: cases.slice(0, 20) };
}

function computeRolling30Day(
	evts: ClassificationEvent[],
	fbMap: Map<string, OutcomeFeedback>,
): RollingConcordance[] {
	if (evts.length === 0) return [];

	// Build daily buckets
	const dailyMap = new Map<string, { concordant: number; total: number }>();

	for (const evt of evts) {
		const fb = fbMap.get(evt.id);
		if (!fb || fb.concordant === undefined) continue;

		const day = evt.timestamp.slice(0, 10); // YYYY-MM-DD
		const entry = dailyMap.get(day) ?? { concordant: 0, total: 0 };
		entry.total++;
		if (fb.concordant) entry.concordant++;
		dailyMap.set(day, entry);
	}

	// Sort days
	const days = Array.from(dailyMap.keys()).sort();
	if (days.length === 0) return [];

	// Compute rolling 30-day window
	const result: RollingConcordance[] = [];
	for (let i = 0; i < days.length; i++) {
		let windowConcordant = 0;
		let windowTotal = 0;

		// Look back up to 30 days
		for (let j = i; j >= 0; j--) {
			const dayA = new Date(days[i]);
			const dayB = new Date(days[j]);
			const diffDays = (dayA.getTime() - dayB.getTime()) / (1000 * 60 * 60 * 24);
			if (diffDays > 30) break;

			const entry = dailyMap.get(days[j])!;
			windowConcordant += entry.concordant;
			windowTotal += entry.total;
		}

		result.push({
			date: days[i],
			concordanceRate: windowTotal > 0 ? windowConcordant / windowTotal : 0,
			n: windowTotal,
		});
	}

	return result;
}
