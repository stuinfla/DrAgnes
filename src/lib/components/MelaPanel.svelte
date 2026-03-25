<script lang="ts">
	import type {
		ClassificationResult,
		ABCDEScores,
		DiagnosisRecord,
		LesionClass,
	} from "$lib/mela/types";
	import { LESION_LABELS } from "$lib/mela/types";
	import { DermClassifier } from "$lib/mela/classifier";
	import { estimateDiameterMm, detectLesionPresence } from "$lib/mela/image-analysis";
	import type {
		ColorAnalysisResult,
		TextureResult,
		StructureResult,
	} from "$lib/mela/image-analysis";
	import { computeTDS, tdsRiskLevel, computeSevenPointScore } from "$lib/mela/clinical-baselines";
	import { getPrimaryICD10, getLocationSpecificICD10 } from "$lib/mela/icd10";
	import { translateForConsumer } from "$lib/mela/consumer-translation";
	import type { ConsumerResult } from "$lib/mela/consumer-translation";
	import { classifyMultiImage } from "$lib/mela/multi-image";
	import type { MultiImageResult } from "$lib/mela/multi-image";
	import { measureLesion } from "$lib/mela/measurement";
	import type { LesionMeasurement } from "$lib/mela/measurement";
	import { applyThresholds } from "$lib/mela/threshold-classifier";
	import type { ThresholdMode } from "$lib/mela/threshold-classifier";
	import { anonymizeCase } from "$lib/mela/anonymization";
	import type { AnonymizedCase } from "$lib/mela/anonymization";
	import { shareToBrain, searchSimilarCases } from "$lib/mela/brain-client";
	import { isOfflineModelLoaded } from "$lib/mela/inference-offline";
	import { warmOfflineModel } from "$lib/mela/inference-orchestrator";
	import type { InferenceStrategy } from "$lib/mela/inference-orchestrator";
	import { ensembleClassify } from "$lib/mela/ensemble";
	import type { EnsembleResult } from "$lib/mela/ensemble";
	import { imageDataToBlob, mapHFResultsToClasses } from "$lib/mela/hf-classifier";
	import { metaClassify } from "$lib/mela/meta-classifier";
	import type { MetaClassification } from "$lib/mela/meta-classifier";

	import DermCapture from "./DermCapture.svelte";
	import GradCamOverlay from "./GradCamOverlay.svelte";
	import LesionTimeline from "./LesionTimeline.svelte";
	import ABCDEChart from "./ABCDEChart.svelte";
	import AnalyticsDashboard from "./AnalyticsDashboard.svelte";
	import ReferralLetter from "./ReferralLetter.svelte";
	import ExplainPanel from "./ExplainPanel.svelte";
	import MethodologyPanel from "./MethodologyPanel.svelte";
	import AboutPage from "./AboutPage.svelte";

	import { recordClassification, recordFeedback } from "$lib/stores/analytics";

	import CarbonCamera from "~icons/carbon/camera";
	import CarbonTime from "~icons/carbon/time";
	import CarbonSettings from "~icons/carbon/settings";
	import CarbonInformation from "~icons/carbon/information";

	const classifier = new DermClassifier();

	/** Svelte action: draw ImageData onto a canvas element */
	function drawImageToCanvas(canvas: HTMLCanvasElement, imageData: ImageData) {
		const ctx = canvas.getContext("2d");
		if (ctx) {
			canvas.width = imageData.width;
			canvas.height = imageData.height;
			ctx.putImageData(imageData, 0, 0);
		}
		return {
			update(newImageData: ImageData) {
				const c = canvas.getContext("2d");
				if (c) {
					canvas.width = newImageData.width;
					canvas.height = newImageData.height;
					c.putImageData(newImageData, 0, 0);
				}
			},
		};
	}

	type ViewId = "scan" | "history" | "learn" | "settings";

	let activeView: ViewId = $state("scan");

	// Capture state
	let capturedImageData: ImageData | null = $state(null);
	let capturedBodyLocation: string = $state("unknown");
	let analyzing: boolean = $state(false);
	let analysisStep: string = $state("");

	// Demographics for HAM10000-calibrated adjustment
	let patientAge: number | undefined = $state(undefined);
	let patientSex: "male" | "female" | undefined = $state(undefined);
	let demographicsEnabled: boolean = $state(true);

	// Clinical history (ADR-130, Dr. Chang feedback)
	let showClinicalHistory: boolean = $state(false);
	let clinicalIsNew: "new" | "months" | "years" | "unsure" = $state("unsure");
	let clinicalHasChanged: "yes" | "no" | "unsure" = $state("unsure");
	let clinicalPreviouslyBiopsied: "yes" | "no" = $state("no");
	let clinicalFamilyHistory: "yes" | "no" | "unsure" = $state("unsure");
	let clinicalSymptoms: ("itching" | "bleeding" | "pain" | "none")[] = $state(["none"]);

	// Results state
	let classificationResult: (ClassificationResult & {
		rawProbabilities?: unknown;
		demographicAdjusted?: boolean;
		clinicalRecommendation?: {
			recommendation: "biopsy" | "urgent_referral" | "monitor" | "reassurance";
			malignantProbability: number;
			melanomaProbability: number;
			reasoning: string;
		};
	}) | null = $state(null);
	let abcdeScores: ABCDEScores | null = $state(null);
	let abcdeEstimated: boolean = $state(true);
	let gradCamData: Uint8Array | null = $state(null);
	let classificationError: string | null = $state(null);
	let lowConfidenceWarning: string | null = $state(null);
	let sevenPointResult: { score: number; recommendation: string; details: string[] } | null = $state(null);
	let lesionMeasurement: LesionMeasurement | null = $state(null);

	// Referral letter modal state
	let showReferralLetter: boolean = $state(false);

	// Demographics panel collapsed on mobile
	let showDemographics: boolean = $state(false);

	// Full-screen image viewer
	let showFullImage: boolean = $state(false);

	// Multi-image capture state
	let multiImageMode: boolean = $state(true); // default ON — better accuracy
	let multiImageCount: number = $state(0);
	let multiImageResult: MultiImageResult | null = $state(null);

	// Explainability findings from last classification
	interface Finding {
		feature: string;
		value: string;
		impact: "supports" | "opposes" | "neutral";
		weight: "strong" | "moderate" | "weak";
		citation: string;
		clinicalSignificance: string;
		tdsWeight: string;
	}
	let explanationFindings: Finding[] = $state([]);

	// History state
	let records: DiagnosisRecord[] = $state([]);

	// Settings state
	let modelVersion: string = $state("v1.0.0-demo");
	let brainSyncEnabled: boolean = $state(false);
	let privacyStripExif: boolean = $state(true);
	let privacyLocalOnly: boolean = $state(true);
	let thresholdMode: ThresholdMode = $state("triage");

	// Inference strategy (ADR-122 Phase 5)
	let inferenceStrategy: InferenceStrategy = $state("auto");
	let offlineModelReady: boolean = $state(false);
	let offlineModelLoading: boolean = $state(false);

	// Offline indicator
	let isOffline: boolean = $state(false);

	// Custom model status
	let customModelAvailable: boolean | null = $state(null);

	// Track the last event ID for feedback binding
	let lastEventId: string | null = $state(null);

	// Outcome feedback state
	let feedbackRecorded: string | null = $state(null);
	let showPathologyInput: boolean = $state(false);
	let showCorrectDropdown: boolean = $state(false);
	let pathologyClass: LesionClass | "" = $state("");
	const ALL_CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

	// ADR-125: V1+V2 ensemble mode
	let ensembleResult: EnsembleResult | null = $state(null);
	let ensembleEnabled: boolean = $state(false);

	// Meta-classifier: neural + clinical feature fusion
	let metaResult: MetaClassification | null = $state(null);

	// Phase 3-4: Network sharing and similar case retrieval
	let caseShared: boolean = $state(false);
	let sharingCase: boolean = $state(false);
	let similarCases: AnonymizedCase[] = $state([]);
	let networkEnabled: boolean = $state(true);

	function handleFeedback(opts: {
		concordant: boolean;
		discordanceReason?: "overcalled" | "missed" | "artifact" | "edge_case" | "other";
		biopsied: boolean;
		malignant?: boolean;
		pathologyResult?: string;
	}) {
		if (!lastEventId) return;
		recordFeedback({
			eventId: lastEventId,
			...opts,
		});
		feedbackRecorded = opts.concordant ? "Agreed" :
			opts.discordanceReason === "overcalled" ? "Overcalled" :
			opts.discordanceReason === "missed" ? "Missed" :
			opts.pathologyResult ? "Pathology recorded" :
			"Feedback recorded";
	}

	/** Load the offline ONNX model on demand. */
	async function loadOfflineModel() {
		if (offlineModelReady || offlineModelLoading) return;
		offlineModelLoading = true;
		offlineModelReady = await warmOfflineModel();
		offlineModelLoading = false;
	}

	function submitPathology() {
		if (!lastEventId || !pathologyClass || !classificationResult) return;
		const isMalignant = pathologyClass === "mel" || pathologyClass === "bcc" || pathologyClass === "akiec";
		const isCorrect = pathologyClass === classificationResult.topClass;
		recordFeedback({
			eventId: lastEventId,
			concordant: isCorrect,
			biopsied: true,
			malignant: isMalignant,
			pathologyResult: pathologyClass,
		});
		feedbackRecorded = "Pathology: " + LESION_LABELS[pathologyClass];
		showPathologyInput = false;
	}

	/** Phase 3: Share anonymized case to pi-brain collective */
	async function shareCase() {
		if (!classificationResult || sharingCase || caseShared) return;
		sharingCase = true;
		try {
			const anonCase = anonymizeCase(
				{
					topClass: classificationResult.topClass,
					confidence: classificationResult.confidence,
					probabilities: classificationResult.probabilities,
				},
				{
					age: patientAge,
					sex: patientSex,
					bodyLocation: capturedBodyLocation,
				},
				{
					diameterMm: abcdeScores?.diameterMm,
					abcdeScore: abcdeScores?.totalScore,
				},
				feedbackRecorded
					? {
							concordant: feedbackRecorded === "Agreed",
							biopsied: feedbackRecorded.startsWith("Pathology"),
							pathologyResult: pathologyClass || undefined,
						}
					: null
			);
			const ok = await shareToBrain(anonCase);
			caseShared = ok;
		} catch {
			// Brain sharing is never a hard dependency
			caseShared = false;
		} finally {
			sharingCase = false;
		}
	}

	/** Phase 4: Fetch similar cases from pi-brain after classification */
	async function fetchSimilarCases() {
		if (!classificationResult || !networkEnabled) return;
		try {
			const probMap: Record<string, number> = {};
			for (const p of classificationResult.probabilities) {
				probMap[p.className] = p.probability;
			}
			const results = await searchSimilarCases(probMap, 5);
			similarCases = results;
		} catch {
			similarCases = [];
		}
	}

	// Scroll container ref
	let scrollContainer: HTMLElement | undefined = $state(undefined);

	// DermCapture ref for reset
	let dermCaptureRef: DermCapture | undefined = $state(undefined);

	$effect(() => {
		if (typeof window !== "undefined") {
			isOffline = !navigator.onLine;
			const handleOnline = () => (isOffline = false);
			const handleOffline = () => (isOffline = true);
			window.addEventListener("online", handleOnline);
			window.addEventListener("offline", handleOffline);
			return () => {
				window.removeEventListener("online", handleOnline);
				window.removeEventListener("offline", handleOffline);
			};
		}
	});

	// Check custom model availability on mount
	$effect(() => {
		if (typeof window !== "undefined") {
			fetch("/api/classify-local")
				.then((res) => res.json())
				.then((data) => {
					customModelAvailable = data.available === true;
				})
				.catch(() => {
					customModelAvailable = false;
				});
		}
	});

	const ANALYSIS_STEPS = [
		"Segmenting lesion...",
		"Analyzing features...",
		"Running neural network...",
		"Generating recommendation...",
	];

	// Bottom nav items -- simplified to 4
	const NAV_ITEMS: { id: ViewId; label: string; icon: typeof CarbonCamera }[] = [
		{ id: "scan", label: "Scan", icon: CarbonCamera },
		{ id: "history", label: "History", icon: CarbonTime },
		{ id: "learn", label: "Learn", icon: CarbonInformation },
		{ id: "settings", label: "Settings", icon: CarbonSettings },
	];

	// Derived: sorted probabilities for bar chart
	let sortedProbabilities = $derived.by(() => {
		const r = classificationResult;
		if (!r) return [];
		return [...r.probabilities].sort((a, b) => b.probability - a.probability);
	});

	// Derived: ICD-10 code for top class
	let icd10 = $derived.by(() => {
		const r = classificationResult;
		return r ? getPrimaryICD10(r.topClass) : null;
	});
	let icd10Location = $derived.by(() => {
		const r = classificationResult;
		return r ? getLocationSpecificICD10(r.topClass, capturedBodyLocation) : null;
	});

	// Derived: recommendation shorthand
	let recommendation = $derived.by(() => {
		return classificationResult?.clinicalRecommendation?.recommendation;
	});

	// Derived: consumer-friendly translation
	let consumerResult: ConsumerResult | null = $derived.by(() => {
		const r = classificationResult;
		if (!r) return null;
		return translateForConsumer(r.topClass, r.confidence, r.probabilities);
	});

	// Derived: whether to show medical details (collapsible)
	let showMedicalDetails: boolean = $state(false);

	// Derived: confidence color utility
	function confidenceTextColor(confidence: number): string {
		if (confidence >= 0.85) return "text-emerald-400";
		if (confidence >= 0.7) return "text-amber-400";
		return "text-red-400";
	}

	function riskBadgeClasses(level: string): string {
		const map: Record<string, string> = {
			low: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
			moderate: "bg-amber-500/20 text-amber-400 border border-amber-500/30",
			high: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
			critical: "bg-red-500/20 text-red-400 border border-red-500/30",
		};
		return map[level] || map.low;
	}

	// ABCDE items for compact row
	let abcdeItems = $derived.by(() => {
		const s = abcdeScores;
		if (!s) return [];
		return [
			{ label: "Asym", value: String(s.asymmetry), max: 2, color: s.asymmetry >= 1 ? "text-red-400" : "text-emerald-400" },
			{ label: "Border", value: String(s.border), max: 8, color: s.border >= 4 ? "text-red-400" : s.border >= 2 ? "text-amber-400" : "text-emerald-400" },
			{ label: "Color", value: String(s.color), max: 6, color: s.color >= 4 ? "text-red-400" : s.color >= 3 ? "text-amber-400" : "text-emerald-400" },
			{ label: "Diam", value: s.diameterMm.toFixed(1) + "mm", max: 10, color: s.diameterMm > 6 ? "text-red-400" : "text-emerald-400" },
			{ label: "Evol", value: String(s.evolution), max: 2, color: s.evolution > 0 ? "text-red-400" : "text-gray-500" },
		];
	});

	function handleCapture(event: { imageData: ImageData; bodyLocation: string; deviceModel: string }) {
		capturedImageData = event.imageData;
		capturedBodyLocation = event.bodyLocation;
		// Auto-analyze immediately — no extra tap needed
		analyzeImage();
	}

	/** Handle multi-image capture — runs consensus classification on all images */
	function handleMultiCapture(events: Array<{ imageData: ImageData; bodyLocation: string; deviceModel: string; imageType: string }>) {
		if (events.length === 0) return;
		capturedImageData = events[0].imageData; // Use first for preview
		capturedBodyLocation = events[0].bodyLocation;
		multiImageCount = events.length;
		analyzeMultiImage(events.map((e) => e.imageData));
	}

	async function analyzeMultiImage(images: ImageData[]) {
		analyzing = true;
		classificationError = null;
		lowConfidenceWarning = null;

		// SAFETY GATE: Check first image for lesion presence
		const lesionCheck = detectLesionPresence(images[0]);
		if (!lesionCheck.hasLesion) {
			classificationError = "healthy_skin:" + lesionCheck.reason;
			analyzing = false;
			return;
		}

		const MULTI_STEPS = [
			`Analyzing image 1 of ${images.length}...`,
			...images.slice(1).map((_, i) => `Analyzing image ${i + 2} of ${images.length}...`),
			"Computing consensus...",
			"Generating recommendation...",
		];
		analysisStep = MULTI_STEPS[0];
		let stepIndex = 0;
		const stepInterval = setInterval(() => {
			stepIndex++;
			if (stepIndex < MULTI_STEPS.length) analysisStep = MULTI_STEPS[stepIndex];
		}, 2500);

		try {
			const demographics = demographicsEnabled
				? { age: patientAge, sex: patientSex, localization: capturedBodyLocation }
				: undefined;
			const result = await classifyMultiImage(classifier, images, demographics);
			multiImageResult = result;

			// Apply per-class thresholds (ADR-123) to adjust top class selection
			if (thresholdMode !== "default") {
				const adjusted = applyThresholds(result.probabilities, thresholdMode);
				classificationResult = {
					...result,
					topClass: adjusted.topClass,
					confidence: adjusted.confidence,
					probabilities: adjusted.probabilities,
				};
			} else {
				classificationResult = result;
			}

			// Use the first image's analysis for ABCDE/explanation (best quality)
			const bestIdx = result.qualityScores.reduce((best, q, i) =>
				q.overallScore > result.qualityScores[best].overallScore ? i : best, 0);

			// Re-run analysis on best image to populate cached features
			await classifier.classifyWithDemographics(images[bestIdx], demographics);

			const realAsymmetry = classifier.getLastAsymmetry();
			const realBorder = classifier.getLastBorderScore();
			const realColors = classifier.getLastColorAnalysis();
			const realSegmentation = classifier.getLastSegmentation();
			const diameterMm = realSegmentation && images[bestIdx]
				? estimateDiameterMm(realSegmentation.area, images[bestIdx].width) : 0;
			const colorCount = realColors ? realColors.colorCount : 1;
			const colorsDetected = realColors ? realColors.dominantColors.map((c) => c.name) : ["light-brown"];
			const structureScore = diameterMm > 6 ? 1 : 0;
			const tds = computeTDS(realAsymmetry, realBorder, colorCount, structureScore);

			abcdeEstimated = false;
			abcdeScores = {
				asymmetry: realAsymmetry,
				border: realBorder,
				color: colorCount,
				diameterMm,
				evolution: 0,
				totalScore: Math.round(tds * 100) / 100,
				riskLevel: tdsRiskLevel(tds),
				colorsDetected,
			};

			// Measure lesion size using ADR-121 measurement system
			if (realSegmentation && images[bestIdx]) {
				lesionMeasurement = await measureLesion(images[bestIdx], realSegmentation.area, capturedBodyLocation as any);
			}

			const structures = classifier.getLastStructures();
			sevenPointResult = structures ? computeSevenPointScore(structures) : null;
			explanationFindings = buildExplanation();

			// Meta-classifier: fuse neural output with clinical features
			metaResult = metaClassify(
				classificationResult!.probabilities,
				abcdeScores,
				abcdeScores?.totalScore ?? null,
				sevenPointResult?.score ?? null,
				abcdeScores?.diameterMm ?? null,
			);

			// Apply meta-classifier adjustments
			classificationResult = {
				...classificationResult!,
				topClass: metaResult.adjustedTopClass,
				confidence: metaResult.adjustedConfidence,
				probabilities: metaResult.adjustedProbabilities,
			};

			// Record in analytics
			const eventId = crypto.randomUUID();
			lastEventId = eventId;
			recordClassification({
				eventId,
				topClass: classificationResult.topClass,
				confidence: classificationResult.confidence,
				probabilities: classificationResult.probabilities,
				bodyLocation: capturedBodyLocation,
				modelId: classificationResult.modelId + ` (${images.length}-image consensus)`,
			});

			if (result.confidence < 0.5) {
				lowConfidenceWarning = "Mela could not classify this with high confidence. This can happen with image quality, unusual lesion appearance, or flesh-colored spots. We recommend seeing a dermatologist to be safe.";
			}

			// Phase 4: Fetch similar cases from pi-brain (non-blocking)
			fetchSimilarCases();
		} catch (err) {
			classificationError = err instanceof Error ? err.message : "Classification failed";
		} finally {
			clearInterval(stepInterval);
			analyzing = false;
		}
	}

	/**
	 * Build explanation findings from cached classifier analysis results.
	 */
	function buildExplanation(): Finding[] {
		const findings: Finding[] = [];
		const asym = classifier.getLastAsymmetry();
		const border = classifier.getLastBorderScore();
		const color: ColorAnalysisResult | null = classifier.getLastColorAnalysis();
		const structures: StructureResult | null = classifier.getLastStructures();
		const texture: TextureResult | null = classifier.getLastTexture();

		if (asym !== undefined) {
			findings.push({
				feature: "Asymmetry",
				value: asym >= 1 ? `Score ${asym}/2 -- asymmetric in ${asym} axis` : "Symmetric",
				impact: asym >= 1 ? "supports" : "opposes",
				weight: asym >= 2 ? "strong" : "moderate",
				citation: "ABCD Rule (Stolz et al., 1994)",
				clinicalSignificance: "Melanomas are asymmetric in >90% of cases. A score of 2 (both axes) has the highest predictive value. Benign nevi are typically symmetric.",
				tdsWeight: "Contributes 1.3x weight in the TDS formula (highest of all ABCD criteria).",
			});
		}
		if (border !== undefined) {
			findings.push({
				feature: "Border",
				value: `${border}/8 segments irregular`,
				impact: border >= 4 ? "supports" : "opposes",
				weight: border >= 6 ? "strong" : "moderate",
				citation: "7-point checklist (Argenziano et al., 1998)",
				clinicalSignificance: `${border >= 4 ? "Sharp, abrupt cutoff in multiple octants suggests irregular growth pattern consistent with malignancy." : "Relatively even borders suggest controlled, benign growth."} Evaluated across 8 equal octants of the lesion perimeter.`,
				tdsWeight: "Contributes 0.1x per irregular segment in the TDS formula (max 0.8).",
			});
		}
		if (color) {
			findings.push({
				feature: "Colors",
				value: `${color.colorCount} distinct colors: ${color.dominantColors?.map((c) => c.name).join(", ")}`,
				impact: color.colorCount >= 3 ? "supports" : "opposes",
				weight: color.colorCount >= 5 ? "strong" : "moderate",
				citation: "ABCD Rule -- C criterion",
				clinicalSignificance: `${color.colorCount >= 3 ? "Multiple colors indicate varied melanin depth and distribution, a hallmark of melanoma." : "Limited color palette is more consistent with benign lesions."} The 6 dermoscopic colors are: white, red, light-brown, dark-brown, blue-gray, black.`,
				tdsWeight: "Contributes 0.5x weight in the TDS formula. 5+ colors is considered highly suspicious.",
			});
			if (color.hasBlueWhiteStructures) {
				findings.push({
					feature: "Blue-white veil",
					value: "Detected -- PPV 0.65 for melanoma",
					impact: "supports",
					weight: "strong",
					citation: "7-point checklist major criterion (2 points)",
					clinicalSignificance: "Blue-white veil indicates dermal melanin and fibrosis (regression). Positive predictive value of 0.65 for melanoma. Considered one of the most specific dermoscopic features.",
					tdsWeight: "Major criterion in 7-point checklist (2 points). Also triggers the melanoma safety gate if combined with another suspicious feature.",
				});
			}
		}
		if (structures) {
			if (structures.hasIrregularNetwork) {
				findings.push({
					feature: "Atypical pigment network",
					value: "Irregular meshes detected",
					impact: "supports",
					weight: "strong",
					citation: "7-point checklist major criterion (2 points)",
					clinicalSignificance: "An irregular pigment network with thick lines and uneven holes suggests abnormal melanocyte proliferation at the dermo-epidermal junction. One of the three major criteria in the 7-point checklist.",
					tdsWeight: "Major criterion (2 points) in the 7-point checklist. A total score of 3 or more triggers biopsy recommendation.",
				});
			}
			if (structures.hasStreaks) {
				findings.push({
					feature: "Streaks/pseudopods",
					value: "Radial structures at periphery",
					impact: "supports",
					weight: "moderate",
					citation: "7-point checklist minor criterion (1 point)",
					clinicalSignificance: "Radial streaks or pseudopods at the lesion periphery indicate radial growth phase -- consistent with early melanoma. Asymmetric distribution increases suspicion.",
					tdsWeight: "Minor criterion (1 point) in the 7-point checklist. Also contributes to the D (differential structures) component of the TDS formula.",
				});
			}
		}
		if (texture) {
			findings.push({
				feature: "Texture",
				value: `Contrast: ${(texture.contrast * 100).toFixed(0)}%, Homogeneity: ${(texture.homogeneity * 100).toFixed(0)}%`,
				impact: texture.homogeneity < 0.4 ? "supports" : "opposes",
				weight: "weak",
				citation: "GLCM texture analysis (Haralick, 1973)",
				clinicalSignificance: `${texture.homogeneity < 0.4 ? "Low textural homogeneity indicates disordered tissue architecture -- associated with melanoma." : "High homogeneity indicates uniform texture, more consistent with benign lesions."} Measured using Gray-Level Co-occurrence Matrix (contrast, homogeneity, entropy).`,
				tdsWeight: "Indirect contributor -- texture disorder increases the D (differential structures) component in the TDS formula.",
			});
		}
		return findings;
	}

	/** ADR-125: Call a single HF model endpoint and return ClassProbability[] */
	async function fetchHFProbabilities(
		endpoint: string,
		blob: Blob,
	): Promise<import("$lib/mela/types").ClassProbability[] | null> {
		try {
			const formData = new FormData();
			formData.append("image", blob, "lesion.jpg");
			const resp = await fetch(endpoint, { method: "POST", body: formData });
			if (!resp.ok) return null;
			const { results } = await resp.json() as {
				results: Array<{ label: string; score: number }>;
			};
			const probMap = mapHFResultsToClasses(results);
			const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
			return CLASSES.map((cls) => ({
				className: cls,
				probability: probMap[cls] ?? 0,
				label: LESION_LABELS[cls],
			}));
		} catch {
			return null;
		}
	}

	async function analyzeImage() {
		if (!capturedImageData) return;
		analyzing = true;
		classificationError = null;
		lowConfidenceWarning = null;
		analysisStep = ANALYSIS_STEPS[0];

		// SAFETY GATE: Check whether the image actually contains a lesion
		const lesionCheck = detectLesionPresence(capturedImageData);

		if (!lesionCheck.hasLesion) {
			classificationError = "healthy_skin:" + lesionCheck.reason;
			analyzing = false;
			return;
		}

		// Cycle through analysis steps for user feedback
		let stepIndex = 0;
		const stepInterval = setInterval(() => {
			stepIndex++;
			if (stepIndex < ANALYSIS_STEPS.length) {
				analysisStep = ANALYSIS_STEPS[stepIndex];
			}
		}, 800);

		try {
			const demographics = demographicsEnabled
				? { age: patientAge, sex: patientSex, localization: capturedBodyLocation }
				: undefined;
			const rawResult = await classifier.classifyWithDemographics(capturedImageData, demographics);

			// Apply per-class thresholds (ADR-123) to adjust top class selection
			if (thresholdMode !== "default") {
				const adjusted = applyThresholds(rawResult.probabilities, thresholdMode);
				classificationResult = {
					...rawResult,
					topClass: adjusted.topClass,
					confidence: adjusted.confidence,
					probabilities: adjusted.probabilities,
				};
			} else {
				classificationResult = rawResult;
			}

			// ADR-125: V1+V2 ensemble override (when enabled)
			ensembleResult = null;
			if (ensembleEnabled && capturedImageData) {
				try {
					const blob = await imageDataToBlob(capturedImageData);
					const [v1Probs, v2Probs] = await Promise.all([
						fetchHFProbabilities("/api/classify", blob),
						fetchHFProbabilities("/api/classify-v2", blob),
					]);
					if (v1Probs && v2Probs) {
						const ens = ensembleClassify(v1Probs, v2Probs);
						ensembleResult = ens;
						// Override the displayed classification with ensemble result
						classificationResult = {
							...classificationResult!,
							topClass: ens.topClass,
							confidence: ens.confidence,
							probabilities: ens.probabilities,
							modelId: classificationResult!.modelId + " + ADR-125-ensemble",
						};
					}
				} catch {
					// Ensemble is best-effort; fall back to single-model result
				}
			}

			// Generate Grad-CAM heatmap
			try {
				const gradCamResult = await classifier.getGradCam(classificationResult.topClass);
				const hm = gradCamResult.heatmap;
				const grayscale = new Uint8Array(hm.width * hm.height);
				for (let i = 0; i < grayscale.length; i++) {
					grayscale[i] = hm.data[i * 4];
				}
				gradCamData = grayscale;
			} catch {
				gradCamData = null;
			}

			// Compute real ABCDE scores from the classifier's image analysis
			const realAsymmetry = classifier.getLastAsymmetry();
			const realBorder = classifier.getLastBorderScore();
			const realColors = classifier.getLastColorAnalysis();
			const realSegmentation = classifier.getLastSegmentation();

			const diameterMm = realSegmentation && capturedImageData
				? estimateDiameterMm(realSegmentation.area, capturedImageData.width)
				: 0;

			const colorCount = realColors ? realColors.colorCount : 1;
			const colorsDetected = realColors
				? realColors.dominantColors.map((c) => c.name)
				: ["light-brown"];

			const structureScore = (diameterMm > 6 ? 1 : 0);
			const tds = computeTDS(realAsymmetry, realBorder, colorCount, structureScore);
			const riskLevel = tdsRiskLevel(tds);

			abcdeEstimated = false;
			abcdeScores = {
				asymmetry: realAsymmetry,
				border: realBorder,
				color: colorCount,
				diameterMm,
				evolution: 0,
				totalScore: Math.round(tds * 100) / 100,
				riskLevel,
				colorsDetected,
			};

			// Measure lesion size using ADR-121 measurement system
			if (realSegmentation && capturedImageData) {
				lesionMeasurement = await measureLesion(capturedImageData, realSegmentation.area, capturedBodyLocation as any);
			}

			// 7-point dermoscopy checklist
			const structures = classifier.getLastStructures();
			if (structures) {
				sevenPointResult = computeSevenPointScore(structures);
			} else {
				sevenPointResult = null;
			}

			// Build explainability findings
			explanationFindings = buildExplanation();

			// Meta-classifier: fuse neural output with clinical features
			metaResult = metaClassify(
				classificationResult.probabilities,
				abcdeScores,
				abcdeScores?.totalScore ?? null,
				sevenPointResult?.score ?? null,
				abcdeScores?.diameterMm ?? null,
			);

			// Apply meta-classifier adjustments to the displayed classification
			classificationResult = {
				...classificationResult,
				topClass: metaResult.adjustedTopClass,
				confidence: metaResult.adjustedConfidence,
				probabilities: metaResult.adjustedProbabilities,
			};

			// Low-confidence warning
			if (classificationResult.confidence < 0.4) {
				lowConfidenceWarning = "Mela could not classify this with enough confidence. This may indicate an unusual lesion type, poor image quality, or a flesh-colored spot with low contrast. We recommend seeing a dermatologist to be safe -- some serious conditions are difficult for any AI to detect.";
			} else {
				lowConfidenceWarning = null;
			}

			// Record classification event
			lastEventId = recordClassification({
				predictedClass: classificationResult.topClass,
				confidence: classificationResult.confidence,
				allProbabilities: Object.fromEntries(
					classificationResult.probabilities.map((p) => [p.className, p.probability]),
				),
				modelId: classificationResult.modelId,
				demographics: { age: patientAge, sex: patientSex },
				bodyLocation: capturedBodyLocation,
			});

			// Phase 4: Fetch similar cases from pi-brain (non-blocking)
			fetchSimilarCases();

			// Scroll to results after a brief delay for animation
			setTimeout(() => {
				scrollContainer?.scrollTo({ top: 0, behavior: "smooth" });
			}, 100);
		} catch (err) {
			console.error("Classification failed:", err);
			classificationError = "Analysis could not be completed. Please retake the image and try again.";
		} finally {
			clearInterval(stepInterval);
			analyzing = false;
			analysisStep = "";
		}
	}

	function handleAction(action: string, payload?: unknown) {
		if (action === "refer") {
			showReferralLetter = true;
			return;
		}
		console.log("Mela action:", action, payload);
	}

	function handleNewScan() {
		capturedImageData = null;
		classificationResult = null;
		abcdeScores = null;
		lesionMeasurement = null;
		gradCamData = null;
		classificationError = null;
		lowConfidenceWarning = null;
		sevenPointResult = null;
		explanationFindings = [];
		lastEventId = null;
		feedbackRecorded = null;
		showPathologyInput = false;
		showCorrectDropdown = false;
		pathologyClass = "";
		showDemographics = false;
		showFullImage = false;
		showMedicalDetails = false;
		analysisStep = "";
		ensembleResult = null;
		metaResult = null;
		multiImageResult = null;
		multiImageCount = 0;
		caseShared = false;
		sharingCase = false;
		similarCases = [];

		// Reset DermCapture component
		dermCaptureRef?.resetCapture();

		// Scroll to top
		scrollContainer?.scrollTo({ top: 0, behavior: "smooth" });
	}

	function correctTo(cls: LesionClass) {
		showCorrectDropdown = false;
		handleAction("correct", cls);
	}
</script>

<!-- Full-screen image viewer overlay -->
{#if showFullImage && capturedImageData}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-sm"
		role="dialog"
		aria-label="Full-screen lesion image"
	>
		<button
			onclick={() => (showFullImage = false)}
			class="absolute right-3 top-3 z-10 flex items-center justify-center rounded-full bg-gray-800/80 p-2.5 text-gray-400 hover:text-white transition-colors"
			style="min-height: 44px; min-width: 44px;"
			aria-label="Close full-screen view"
		>
			<svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
			</svg>
		</button>
		<canvas
			class="max-h-[90vh] max-w-[90vw] rounded-2xl object-contain"
			width={capturedImageData.width}
			height={capturedImageData.height}
			use:drawImageToCanvas={capturedImageData}
		></canvas>
	</div>
{/if}

<div class="flex h-full w-full flex-col bg-[#0a0a0f]">
	{#if activeView === "scan"}
		<!-- ===== PRIMARY SCAN FLOW ===== -->
		<div
			bind:this={scrollContainer}
			class="scrollbar-thin flex-1 overflow-y-auto overscroll-none pb-20"
		>
			<!-- Offline banner -->
			{#if isOffline}
				<div class="mx-5 mt-3 flex items-center gap-2.5 rounded-2xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-400">
					<svg class="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414"></path></svg>
					<span>Offline -- Local analysis available</span>
				</div>
			{/if}

			<!-- ===== RESULTS ZONE (when we have a result) ===== -->
			{#if classificationResult && !analyzing}
				<section class="animate-scaleIn">

					<!-- Consumer-Friendly Hero Result -->
					{#if consumerResult}
						<div class="flex flex-col items-center gap-6 px-5 pt-10 pb-8 text-center {
							consumerResult.riskLevel === 'green' ? 'gradient-emerald-glow' :
							consumerResult.riskLevel === 'red' ? 'gradient-red-glow' :
							'gradient-amber-glow'
						}">
							<!-- Big status icon -->
							{#if consumerResult.riskLevel === "green"}
								<div class="animate-scaleIn">
									<img src="/result-safe.png" alt="Low concern" class="h-20 w-20 rounded-2xl object-cover shadow-lg shadow-emerald-500/20" />
								</div>
							{:else if consumerResult.riskLevel === "red"}
								<div class="animate-scaleIn">
									<img src="/result-urgent.png" alt="See a doctor" class="h-20 w-20 rounded-2xl object-cover shadow-lg shadow-red-500/20" />
								</div>
							{:else if consumerResult.riskLevel === "orange"}
								<div class="h-24 w-24 rounded-full bg-orange-500/15 flex items-center justify-center ring-orange animate-scaleIn">
									<svg class="h-12 w-12 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5">
										<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
									</svg>
								</div>
							{:else}
								<div class="h-24 w-24 rounded-full bg-amber-500/15 flex items-center justify-center ring-amber animate-scaleIn">
									<svg class="h-12 w-12 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5">
										<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01"></path>
										<circle cx="12" cy="12" r="10" stroke-width="2"></circle>
									</svg>
								</div>
							{/if}

							<!-- Headline -->
							<div class="animate-countUp" style="animation-delay: 0.1s;">
								<h2 class="text-2xl font-bold tracking-tight {
									consumerResult.riskLevel === 'green' ? 'text-emerald-400' :
									consumerResult.riskLevel === 'red' ? 'text-red-400' :
									consumerResult.riskLevel === 'orange' ? 'text-orange-400' :
									'text-amber-400'
								}">
									{consumerResult.headline}
								</h2>
								<p class="mt-3 text-[15px] text-gray-400 leading-relaxed text-balance">
									{consumerResult.explanation.split('.')[0]}.
								</p>
							</div>

							<!-- Action -->
							<div class="animate-countUp rounded-2xl bg-white/[0.03] border border-white/[0.06] px-6 py-5 w-full max-w-sm" style="animation-delay: 0.2s;">
								<p class="text-[15px] font-medium text-gray-200 leading-relaxed">{consumerResult.action}</p>
							</div>

							<!-- Confidence pill -->
							<div class="animate-countUp flex items-center gap-2.5 text-[11px] text-gray-500" style="animation-delay: 0.3s;">
								<span class="{confidenceTextColor(classificationResult.confidence)} font-semibold">{(classificationResult.confidence * 100).toFixed(0)}% confidence</span>
								<span class="text-gray-700">|</span>
								<span>{Math.round(classificationResult.inferenceTimeMs)}ms</span>
								{#if multiImageResult}
									<span class="text-gray-700">|</span>
									<span class="text-teal-400">{multiImageCount} photos analyzed</span>
								{/if}
							</div>
							{#if multiImageResult}
								<div class="animate-countUp flex items-center gap-1.5 text-[10px] mt-1" style="animation-delay: 0.4s;">
									<span class="inline-block h-1.5 w-1.5 rounded-full {multiImageResult.agreementScore > 0.8 ? 'bg-emerald-500' : multiImageResult.agreementScore > 0.5 ? 'bg-amber-400' : 'bg-red-400'}"></span>
									<span class="text-gray-500">Image agreement: {(multiImageResult.agreementScore * 100).toFixed(0)}%</span>
								</div>
							{/if}
						</div>
					{/if}

					<!-- Captured image (compact, tappable) -->
					{#if capturedImageData}
						<div class="mx-5 mt-5">
							<button
								onclick={() => (showFullImage = true)}
								class="relative block w-full overflow-hidden rounded-2xl bg-gray-900 border border-white/[0.06]"
								aria-label="View full-screen image"
							>
								<canvas
									class="h-[18vh] w-full object-contain"
									width={capturedImageData.width}
									height={capturedImageData.height}
									use:drawImageToCanvas={capturedImageData}
								></canvas>
								<div class="absolute bottom-2 right-2 rounded-full bg-black/60 px-2.5 py-1 text-[10px] text-gray-400 backdrop-blur-sm">
									Tap to expand
								</div>
							</button>
						</div>
					{/if}

					<!-- Model disagreement warning -->
					{#if classificationResult.modelsDisagree}
						<div class="mx-5 mt-3 flex items-center gap-2.5 rounded-2xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-400">
							<svg class="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
							</svg>
							<span>Models disagree -- Agreement: {((classificationResult.modelAgreement || 0) * 100).toFixed(0)}%</span>
						</div>
					{/if}

					<!-- ADR-125 ensemble disagreement warning -->
					{#if ensembleResult && !ensembleResult.modelsAgree}
						<div class="mx-5 mt-3 rounded-2xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-400">
							{ensembleResult.disagreementWarning}
						</div>
					{/if}

					<!-- Meta-classifier: clinical feature agreement note -->
					{#if metaResult && metaResult.agreement !== "neutral"}
						<div class="mx-5 mt-3 flex items-center gap-2.5 rounded-2xl border px-4 py-3 text-xs {metaResult.agreement === 'concordant' ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-400' : 'border-amber-500/20 bg-amber-500/5 text-amber-400'}">
							{#if metaResult.agreement === "concordant"}
								<svg class="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
								</svg>
							{:else}
								<svg class="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
								</svg>
							{/if}
							<span>{metaResult.adjustmentReason}</span>
						</div>
					{/if}

					<!-- Low-confidence warning -->
					{#if lowConfidenceWarning}
						<div class="mx-5 mt-3 flex items-center gap-2.5 rounded-2xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-400">
							<svg class="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
							</svg>
							<span>{lowConfidenceWarning}</span>
						</div>
					{/if}

					<!-- Fitzpatrick equity note -->
					<div class="mx-5 mt-3 rounded-2xl border border-gray-700/40 bg-white/[0.02] px-4 py-2.5">
						<p class="text-[10px] text-gray-500 leading-relaxed">
							Skin tone note: Validation found a 30pp sensitivity gap across Fitzpatrick types. Tested on dermoscopy images, not clinical photos. Dark skin performance unverified.
							<button onclick={() => (activeView = "learn")} class="text-gray-400 underline hover:text-gray-300 transition-colors">Details</button>
						</p>
					</div>

					<!-- Action buttons: clean and spacious -->
					<div class="mx-5 mt-6 grid grid-cols-2 gap-3">
						{#if consumerResult?.shouldSeeDoctor}
							<button
								onclick={() => handleAction("refer")}
								class="flex h-13 items-center justify-center gap-2 rounded-full bg-teal-600 px-6 py-3 text-[15px] font-semibold text-white shadow-lg shadow-teal-600/20 hover:bg-teal-500 active:scale-95 transition-all col-span-2"
							>
								<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
								Generate Referral Letter
							</button>
						{/if}
						<button
							onclick={handleNewScan}
							class="flex h-12 items-center justify-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-6 py-3 text-sm font-medium text-gray-300 hover:bg-white/[0.06] active:scale-95 transition-all {consumerResult?.shouldSeeDoctor ? '' : 'col-span-2'}"
						>
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
							New Scan
						</button>
						{#if consumerResult?.shouldSeeDoctor}
							<button
								onclick={() => handleAction("biopsy")}
								class="flex h-12 items-center justify-center rounded-full border border-white/[0.08] bg-white/[0.03] px-6 py-3 text-sm font-medium text-gray-300 hover:bg-white/[0.06] active:scale-95 transition-all"
							>
								Biopsy
							</button>
						{/if}
					</div>

					<!-- Medical Details (collapsible) -->
					<div class="mx-5 mt-8">
						<button
							onclick={() => (showMedicalDetails = !showMedicalDetails)}
							class="flex w-full items-center justify-between rounded-2xl border border-white/[0.06] bg-white/[0.02] px-4 py-3.5 text-left transition-colors hover:bg-white/[0.04]"
						>
							<span class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Medical Details</span>
							<svg
								class="h-4 w-4 text-gray-500 transition-transform duration-300 {showMedicalDetails ? 'rotate-180' : ''}"
								fill="none" stroke="currentColor" viewBox="0 0 24 24"
							>
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
							</svg>
						</button>

						{#if showMedicalDetails}
							<div class="mt-3 space-y-3 animate-fadeIn">

								<!-- Top prediction technical card -->
								<div class="card">
									<div class="flex items-baseline justify-between mb-3">
										<h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Classification</h3>
										{#if abcdeScores}
											<span class="rounded-full px-2.5 py-0.5 text-[10px] font-semibold {riskBadgeClasses(abcdeScores.riskLevel)}">
												{abcdeScores.riskLevel.toUpperCase()}
											</span>
										{/if}
									</div>
									<p class="text-lg font-bold text-gray-100">
										{LESION_LABELS[classificationResult.topClass]}
									</p>
									<div class="mt-1 flex items-center gap-3 text-[10px] text-gray-500">
										{#if icd10}
											<span class="font-mono">{icd10.code}</span>
										{/if}
										{#if classificationResult.usedDualModel}
											<span class="text-teal-500/70">Dual ViT</span>
										{:else if classificationResult.usedHF}
											<span class="text-teal-500/70">ViT + Local</span>
										{:else if classificationResult.usedWasm}
											<span class="text-amber-500/70">WASM</span>
										{:else}
											<span class="text-amber-500/70">Local only</span>
										{/if}
									</div>

									<!-- Demographic adjustment note -->
									{#if classificationResult.demographicAdjusted}
										<p class="mt-2 text-[10px] italic text-gray-500">Adjusted with HAM10000 demographics</p>
									{/if}
								</div>

								<!-- Biopsy / Urgent referral banner -->
								{#if recommendation === "urgent_referral"}
									<div class="rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-center">
										<p class="text-sm font-bold text-red-300">URGENT: Dermatology Referral Recommended</p>
										{#if classificationResult.clinicalRecommendation}
											<p class="mt-1.5 text-[11px] text-red-400/70 leading-relaxed">{classificationResult.clinicalRecommendation.reasoning}</p>
											<div class="mt-2 flex justify-center gap-4 text-[10px] text-red-400/60">
												<span>Mel P: {(classificationResult.clinicalRecommendation.melanomaProbability * 100).toFixed(1)}%</span>
												<span>Malignant P: {(classificationResult.clinicalRecommendation.malignantProbability * 100).toFixed(1)}%</span>
											</div>
										{/if}
									</div>
								{:else if recommendation === "biopsy"}
									<div class="rounded-2xl border border-orange-500/30 bg-orange-500/10 p-4 text-center">
										<p class="text-sm font-bold text-orange-300">Biopsy Recommended</p>
										{#if classificationResult.clinicalRecommendation}
											<p class="mt-1.5 text-[11px] text-orange-400/70 leading-relaxed">{classificationResult.clinicalRecommendation.reasoning}</p>
										{/if}
									</div>
								{:else if recommendation === "monitor"}
									<div class="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4 text-center">
										<p class="text-sm font-semibold text-amber-300">Monitor -- Follow Up</p>
										{#if classificationResult.clinicalRecommendation}
											<p class="mt-1.5 text-[11px] text-amber-400/60 leading-relaxed">{classificationResult.clinicalRecommendation.reasoning}</p>
										{/if}
									</div>
								{/if}

								<!-- ABCDE compact row -->
								{#if abcdeItems.length > 0}
									<div class="flex gap-2 overflow-x-auto scrollbar-hide py-1" style="-webkit-overflow-scrolling: touch;">
										{#each abcdeItems as item}
											<div class="flex-shrink-0 rounded-2xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5 text-center min-w-[56px]">
												<div class="text-sm font-bold tabular-nums {item.color}">{item.value}</div>
												<div class="text-[8px] text-gray-500 uppercase tracking-wider mt-0.5">{item.label}</div>
											</div>
										{/each}
										{#if abcdeScores}
											<div class="flex-shrink-0 rounded-2xl bg-white/[0.03] border border-white/[0.08] px-3 py-2.5 text-center min-w-[56px]">
												<div class="text-sm font-bold tabular-nums text-gray-200">{abcdeScores.totalScore.toFixed(1)}</div>
												<div class="text-[8px] text-gray-500 uppercase tracking-wider mt-0.5">TDS</div>
											</div>
										{/if}
									</div>
								{/if}

								<!-- Estimated Size (ADR-121 measurement) -->
								{#if lesionMeasurement}
									<div class="mx-5 mt-3 rounded-2xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
										<div class="flex items-center justify-between">
											<span class="text-[11px] font-medium text-gray-400">Estimated Size</span>
											<span class="text-sm font-semibold {lesionMeasurement.diameterMm >= 6 ? 'text-amber-400' : 'text-gray-200'}">
												{lesionMeasurement.diameterMm.toFixed(1)}mm
											</span>
										</div>
										<div class="mt-1 flex items-center gap-1.5">
											<span class="inline-block h-1.5 w-1.5 rounded-full {lesionMeasurement.confidence === 'high' ? 'bg-emerald-500' : lesionMeasurement.confidence === 'medium' ? 'bg-amber-400' : 'bg-gray-500'}"></span>
											<span class="text-[10px] text-gray-500">{lesionMeasurement.details}</span>
										</div>
										{#if lesionMeasurement.diameterMm >= 6}
											<p class="mt-2 text-[10px] text-amber-400/80">Diameter ≥ 6mm — a clinical indicator in the ABCDE criteria.</p>
										{/if}
									</div>
								{/if}

								<!-- Probability bars -->
								<div class="card space-y-2">
									<h4 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Probabilities</h4>
									{#each sortedProbabilities as prob}
										<div class="flex items-center gap-2">
											<span class="text-[10px] text-gray-400 w-12 text-right font-mono shrink-0">{prob.className}</span>
											<div class="flex-1 h-2.5 bg-white/[0.04] rounded-full overflow-hidden">
												<div
													class="h-full rounded-full transition-all duration-500 {prob.className === classificationResult.topClass ? 'bg-teal-500' : 'bg-gray-600/60'}"
													style="width: {Math.max(prob.probability * 100, 0.5)}%"
												></div>
											</div>
											<span class="text-[10px] text-gray-400 w-11 text-right tabular-nums shrink-0">{(prob.probability * 100).toFixed(1)}%</span>
										</div>
									{/each}
								</div>

								<!-- 7-point checklist -->
								{#if sevenPointResult}
									<div class="card">
										<div class="flex items-center gap-3">
											<span class="text-xs text-gray-400">7-Point Checklist:</span>
											<span class="font-bold text-sm {sevenPointResult.score >= 3 ? 'text-red-400' : 'text-gray-300'}">{sevenPointResult.score}</span>
											<span class="text-xs {sevenPointResult.score >= 3 ? 'text-red-400' : 'text-gray-500'}">{sevenPointResult.recommendation}</span>
										</div>
									</div>
								{/if}

								<!-- Grad-CAM attention map -->
								{#if capturedImageData && gradCamData}
									<details class="rounded-2xl border border-white/[0.06] overflow-hidden">
										<summary class="px-4 py-3.5 text-xs text-gray-400 cursor-pointer hover:text-gray-300 transition-colors touch-target font-medium">
											Attention Map (Grad-CAM)
										</summary>
										<div class="px-4 pb-4">
											<GradCamOverlay imageData={capturedImageData} gradCamData={gradCamData} />
										</div>
									</details>
								{/if}

								<!-- ABCDE Radar Chart -->
								{#if abcdeScores}
									<details class="rounded-2xl border border-white/[0.06] overflow-hidden">
										<summary class="px-4 py-3.5 text-xs text-gray-400 cursor-pointer hover:text-gray-300 transition-colors touch-target font-medium">
											ABCDE Radar Chart
										</summary>
										<div class="px-4 pb-4">
											<ABCDEChart scores={abcdeScores} />
										</div>
									</details>
								{/if}

								<!-- Explainability -->
								{#if explanationFindings.length > 0}
									<details class="rounded-2xl border border-white/[0.06] overflow-hidden">
										<summary class="px-4 py-3.5 text-xs text-gray-400 cursor-pointer hover:text-gray-300 transition-colors touch-target font-medium">
											Why this classification?
										</summary>
										<div class="px-4 pb-4">
											<ExplainPanel
												topClass={LESION_LABELS[classificationResult.topClass]}
												findings={explanationFindings}
											/>
										</div>
									</details>
								{/if}

								<!-- ICD-10 & Referral -->
								{#if icd10}
									<details class="rounded-2xl border border-white/[0.06] overflow-hidden">
										<summary class="px-4 py-3.5 text-xs text-gray-400 cursor-pointer hover:text-gray-300 transition-colors touch-target font-medium">
											ICD-10 & Referral
										</summary>
										<div class="px-4 pb-4 space-y-2">
											<div class="flex items-center gap-2 text-xs">
												<span class="text-gray-500">Primary:</span>
												<span class="font-mono text-gray-300">{icd10.code}</span>
												<span class="text-gray-400">{icd10.description}</span>
											</div>
											{#if icd10Location && icd10Location.code !== icd10.code}
												<div class="flex items-center gap-2 text-xs">
													<span class="text-gray-500">Location:</span>
													<span class="font-mono text-gray-300">{icd10Location.code}</span>
													<span class="text-gray-400">{icd10Location.description}</span>
												</div>
											{/if}
											<div class="text-[10px] text-gray-500">
												Category: <span class="{icd10.category === 'malignant' ? 'text-red-400' : icd10.category === 'in_situ' ? 'text-orange-400' : 'text-emerald-400'}">{icd10.category}</span>
											</div>
										</div>
									</details>
								{/if}

								<!-- Model Provenance -->
								<details class="rounded-2xl border border-white/[0.06] overflow-hidden">
									<summary class="px-4 py-3.5 text-xs text-gray-400 cursor-pointer hover:text-gray-300 transition-colors touch-target font-medium">
										Model Provenance
									</summary>
									<div class="px-4 pb-4 space-y-2">
										<div class="rounded-xl bg-white/[0.03] p-3 text-[10px]">
											<p class="font-medium text-gray-300 mb-1">Classification Mode</p>
											{#if classificationResult.usedDualModel}
												<p class="text-teal-400">Dual-ViT Ensemble -- Highest accuracy mode</p>
											{:else if classificationResult.usedHF}
												<p class="text-teal-400">Single-ViT + Local Analysis</p>
											{:else if classificationResult.usedWasm}
												<p class="text-amber-400">WASM CNN (local)</p>
											{:else}
												<p class="text-amber-400">Local Analysis Only (offline)</p>
											{/if}
										</div>
										<p class="text-[10px] font-mono text-gray-600">Model ID: {classificationResult.modelId}</p>
									</div>
								</details>

								<!-- Outcome Feedback -->
								{#if lastEventId}
									<div class="card">
										{#if feedbackRecorded}
											<div class="flex items-center gap-2 text-xs text-teal-400">
												<svg class="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
												<span>{feedbackRecorded}</span>
												<button
													onclick={() => (feedbackRecorded = null)}
													class="ml-auto text-gray-500 hover:text-gray-400 text-[10px]"
												>
													Change
												</button>
											</div>
										{:else}
											<h4 class="text-[10px] font-medium text-gray-500 mb-3 uppercase tracking-wider">Record Outcome</h4>
											<div class="flex gap-2 flex-wrap">
												<button
													class="rounded-xl border border-teal-500/30 bg-teal-500/10 px-3 py-2.5 text-[11px] text-teal-400 hover:bg-teal-500/20 active:bg-teal-500/30 transition-colors touch-target"
													onclick={() => handleFeedback({ concordant: true, biopsied: false })}
												>
													AI Agreed
												</button>
												<button
													class="rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-[11px] text-amber-400 hover:bg-amber-500/20 active:bg-amber-500/30 transition-colors touch-target"
													onclick={() => handleFeedback({ concordant: false, discordanceReason: "overcalled", biopsied: false })}
												>
													Overcalled
												</button>
												<button
													class="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2.5 text-[11px] text-red-400 hover:bg-red-500/20 active:bg-red-500/30 transition-colors touch-target"
													onclick={() => handleFeedback({ concordant: false, discordanceReason: "missed", biopsied: true })}
												>
													AI Missed
												</button>
												<button
													class="rounded-xl border border-blue-500/30 bg-blue-500/10 px-3 py-2.5 text-[11px] text-blue-400 hover:bg-blue-500/20 active:bg-blue-500/30 transition-colors touch-target"
													onclick={() => (showPathologyInput = true)}
												>
													Pathology
												</button>
											</div>

											{#if showPathologyInput}
												<div class="mt-3 flex items-end gap-2">
													<div class="flex-1">
														<label class="mb-1 block text-[10px] text-gray-500" for="pathology-select-panel">Result</label>
														<select
															id="pathology-select-panel"
															bind:value={pathologyClass}
															class="w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-xs text-gray-200"
														>
															<option value="">Select...</option>
															{#each ALL_CLASSES as cls}
																<option value={cls}>{cls} -- {LESION_LABELS[cls]}</option>
															{/each}
														</select>
													</div>
													<button
														onclick={submitPathology}
														disabled={!pathologyClass}
														class="rounded-xl bg-blue-600 px-4 py-2 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-40"
													>
														Save
													</button>
													<button
														onclick={() => (showPathologyInput = false)}
														class="rounded-xl bg-white/[0.05] px-4 py-2 text-xs text-gray-300 hover:bg-white/[0.08]"
													>
														Cancel
													</button>
												</div>
											{/if}
										{/if}
									</div>
								{/if}
							</div>
						{/if}
					</div>

					<!-- Phase 3: Share anonymized case to pi-brain -->
					{#if networkEnabled && classificationResult}
						<div class="mx-5 mt-4 rounded-2xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
							<div class="flex items-center justify-between">
								<span class="text-[11px] font-medium text-gray-400">Share anonymized case</span>
								{#if caseShared}
									<span class="text-[10px] text-emerald-400">Shared successfully</span>
								{:else}
									<button
										onclick={shareCase}
										disabled={sharingCase}
										class="text-[10px] text-teal-400 hover:text-teal-300 disabled:opacity-50 disabled:cursor-wait transition-colors"
									>
										{sharingCase ? "Sharing..." : "Share to network"}
									</button>
								{/if}
							</div>
							<p class="text-[9px] text-gray-600 mt-1">
								Anonymized classification data shared via pi-brain. No images leave your device.
							</p>
						</div>
					{/if}

					<!-- Phase 4: Similar cases from pi-brain -->
					{#if similarCases.length > 0}
						<div class="mx-5 mt-3 rounded-2xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
							<h4 class="text-[11px] font-medium text-gray-400 mb-2">Similar cases from the network</h4>
							{#each similarCases.slice(0, 3) as sc}
								<div class="flex items-center justify-between py-1">
									<span class="text-[10px] text-gray-400">{sc.topClass}</span>
									<span class="text-[10px] text-gray-500">{sc.outcome || "No outcome recorded"}</span>
								</div>
							{/each}
							{#if similarCases.length > 3}
								<p class="text-[9px] text-gray-600 mt-1">+{similarCases.length - 3} more similar cases</p>
							{/if}
						</div>
					{/if}

					<!-- Research disclaimer -->
					<p class="mx-5 mt-8 mb-8 text-[11px] text-gray-600 italic text-center leading-relaxed">
						AI-generated suggestion -- clinical judgment must supersede. Not FDA-cleared.
					</p>
				</section>

			<!-- ===== ANALYZING STATE ===== -->
			{:else if analyzing}
				<div class="flex flex-col items-center justify-center gap-5 py-16 animate-fadeIn">
					<div class="relative h-16 w-16">
						<div class="absolute inset-0 rounded-full border-2 border-teal-500/20 animate-ping"></div>
						<div class="absolute inset-2 rounded-full border-2 border-teal-500 animate-spin border-t-transparent"></div>
						<div class="absolute inset-4 rounded-full border-2 border-teal-400/40 animate-spin border-b-transparent" style="animation-direction: reverse; animation-duration: 1.5s;"></div>
					</div>
					<p class="text-sm text-teal-400 animate-pulse text-center px-6">{analysisStep}</p>
				</div>

			<!-- ===== ERROR STATE ===== -->
			{:else if classificationError && classificationError.startsWith("healthy_skin:")}
				<!-- Healthy skin: no concerning features -->
				<div class="flex flex-col items-center gap-5 px-6 py-12 text-center animate-fadeIn">
					<div class="h-20 w-20 rounded-full bg-emerald-500/15 flex items-center justify-center">
						<svg class="h-10 w-10 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
							<path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
						</svg>
					</div>
					<div>
						<h2 class="text-xl font-semibold text-emerald-300">{classificationError.replace("healthy_skin:", "Your skin looks healthy! ")}</h2>
						<p class="mt-2 text-sm text-gray-400 max-w-xs mx-auto">To check a specific area, photograph a mole or spot you want evaluated.</p>
					</div>
					<button
						onclick={handleNewScan}
						class="rounded-full bg-teal-600 px-6 py-3 text-sm font-medium text-white hover:bg-teal-500 active:scale-95 transition-all"
					>
						Scan a Spot
					</button>
				</div>

			{:else if classificationError}
				<!-- Generic classification error -->
				<div class="flex flex-col items-center gap-5 px-6 py-12 text-center animate-fadeIn">
					<div class="h-20 w-20 rounded-full bg-red-500/15 flex items-center justify-center ring-red">
						<svg class="h-10 w-10 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
							<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path>
						</svg>
					</div>
					<p class="text-sm font-medium text-red-400">{classificationError.replace("healthy_skin:", "Your skin looks healthy! ")}</p>
					<button
						onclick={handleNewScan}
						class="rounded-full bg-red-500/20 px-6 py-3 text-sm font-medium text-red-300 hover:bg-red-500/30 active:scale-95 transition-all"
					>
						Retake Image
					</button>
				</div>

			<!-- ===== CAPTURE ZONE (default / initial state) ===== -->
			{:else if !capturedImageData}
				<!-- Trust banner (above camera) -->
				<div class="mx-5 mt-4 rounded-2xl bg-teal-500/5 border border-teal-500/15 px-5 py-3.5 text-center">
					<p class="text-[11px] text-teal-400/80">
						95.97% melanoma detection on external data &bull; 37,484 training images &bull;
						<a href="/how-it-works.html" target="_blank" class="underline hover:text-teal-300 transition-colors">How it works</a>
					</p>
				</div>

				<!-- Inviting empty state -->
				<div class="flex flex-col items-center justify-center gap-6 px-5 py-8 text-center animate-fadeIn">
					<img
						src="/hero-scan.png"
						alt="Dermatoscope lens examining skin"
						class="h-24 w-24 rounded-full object-cover shadow-lg shadow-teal-500/20 ring-2 ring-teal-500/20 animate-pulseGlow"
					/>
					<div>
						<h2 class="text-2xl font-semibold text-white leading-tight">Check a skin spot</h2>
						<p class="mt-2.5 text-[15px] text-gray-400 leading-relaxed max-w-xs mx-auto">Take a photo of what concerns you. We'll analyze it in seconds.</p>
					</div>
				</div>

				<!-- Photo guidance tips -->
				<div class="mx-5 mb-3 grid grid-cols-2 gap-2">
					<div class="flex items-center gap-2 rounded-xl bg-white/[0.03] px-3 py-2">
						<span class="text-amber-400 text-sm">&#9728;</span>
						<span class="text-[10px] text-gray-400">Good lighting, no shadows</span>
					</div>
					<div class="flex items-center gap-2 rounded-xl bg-white/[0.03] px-3 py-2">
						<span class="text-teal-400 text-sm">&#8982;</span>
						<span class="text-[10px] text-gray-400">4-6 inches from the spot</span>
					</div>
					<div class="flex items-center gap-2 rounded-xl bg-white/[0.03] px-3 py-2">
						<span class="text-blue-400 text-sm">&#9673;</span>
						<span class="text-[10px] text-gray-400">One spot per photo</span>
					</div>
					<div class="flex items-center gap-2 rounded-xl bg-white/[0.03] px-3 py-2">
						<span class="text-purple-400 text-sm">&#10023;</span>
						<span class="text-[10px] text-gray-400">Clean camera lens</span>
					</div>
				</div>

				<!-- Full capture area -->
				<div class="px-0 sm:px-4 lg:max-w-[600px] lg:mx-auto">
					<DermCapture
						oncapture={handleCapture}
						multiCapture={multiImageMode}
						maxImages={3}
						onmulticapture={handleMultiCapture}
						bind:this={dermCaptureRef}
					/>
				</div>

				<!-- Demographics toggle -->
				<div class="mx-5 mt-4">
					<button
						onclick={() => (showDemographics = !showDemographics)}
						class="flex w-full items-center justify-between rounded-2xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-left transition-colors hover:bg-white/[0.04] touch-target"
					>
						<span class="text-[11px] font-medium text-gray-400">Patient Demographics</span>
						<svg
							class="h-4 w-4 text-gray-500 transition-transform duration-300 {showDemographics ? 'rotate-180' : ''}"
							fill="none" stroke="currentColor" viewBox="0 0 24 24"
						>
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
						</svg>
					</button>
					{#if showDemographics}
						<div class="mt-2 rounded-2xl border border-white/[0.06] bg-white/[0.02] p-4 animate-fadeIn">
							<div class="mb-3 flex items-center justify-end">
								<label class="flex items-center gap-2">
									<span class="text-[11px] text-gray-500">Enable</span>
									<input
										type="checkbox"
										bind:checked={demographicsEnabled}
										class="h-4 w-4 rounded border-gray-600 bg-gray-800 text-teal-500"
									/>
								</label>
							</div>
							{#if demographicsEnabled}
								<div class="grid grid-cols-2 gap-3">
									<div>
										<label class="mb-1 block text-[11px] text-gray-500" for="age-input">Age</label>
										<input
											id="age-input"
											type="number"
											min="0"
											max="120"
											placeholder="e.g. 55"
											bind:value={patientAge}
											class="w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-3 py-2.5 text-sm text-gray-200"
										/>
									</div>
									<div>
										<label class="mb-1 block text-[11px] text-gray-500" for="sex-input">Sex</label>
										<select
											id="sex-input"
											bind:value={patientSex}
											class="w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-3 py-2.5 text-sm text-gray-200"
										>
											<option value={undefined}>Not specified</option>
											<option value="male">Male</option>
											<option value="female">Female</option>
										</select>
									</div>
								</div>
							{/if}
						</div>
					{/if}
				</div>

				<!-- Clinical context (ADR-130, Dr. Chang) -->
				<div class="mx-5 mt-3">
					<button
						onclick={() => (showClinicalHistory = !showClinicalHistory)}
						class="flex w-full items-center justify-between rounded-2xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-left transition-colors hover:bg-white/[0.04] touch-target"
					>
						<span class="text-[11px] font-medium text-gray-400">Clinical Context (optional)</span>
						<svg
							class="h-4 w-4 text-gray-500 transition-transform duration-300 {showClinicalHistory ? 'rotate-180' : ''}"
							fill="none" stroke="currentColor" viewBox="0 0 24 24"
						>
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
						</svg>
					</button>
					{#if showClinicalHistory}
						<div class="mt-2 rounded-2xl border border-white/[0.06] bg-white/[0.02] p-4 animate-fadeIn">
							<p class="text-[10px] text-gray-500 mb-3">These questions help Mela give you a more accurate risk assessment.</p>

							<div class="flex flex-col gap-3">
								<div>
									<label class="mb-1 block text-[11px] text-gray-400">Is this spot new?</label>
									<div class="flex gap-2">
										{#each [["new", "New"], ["months", "Months"], ["years", "Years"], ["unsure", "Not sure"]] as [val, label]}
											<button
												onclick={() => clinicalIsNew = val as any}
												class="flex-1 rounded-lg px-2 py-1.5 text-[10px] border transition-colors {clinicalIsNew === val ? 'bg-teal-500/20 border-teal-500/40 text-teal-300' : 'bg-white/[0.03] border-white/[0.06] text-gray-400'}"
											>{label}</button>
										{/each}
									</div>
								</div>

								<div>
									<label class="mb-1 block text-[11px] text-gray-400">Has it changed recently?</label>
									<div class="flex gap-2">
										{#each [["yes", "Yes"], ["no", "No"], ["unsure", "Not sure"]] as [val, label]}
											<button
												onclick={() => clinicalHasChanged = val as any}
												class="flex-1 rounded-lg px-2 py-1.5 text-[10px] border transition-colors {clinicalHasChanged === val ? 'bg-teal-500/20 border-teal-500/40 text-teal-300' : 'bg-white/[0.03] border-white/[0.06] text-gray-400'}"
											>{label}</button>
										{/each}
									</div>
								</div>

								<div>
									<label class="mb-1 block text-[11px] text-gray-400">Previously biopsied?</label>
									<div class="flex gap-2">
										{#each [["no", "No"], ["yes", "Yes"]] as [val, label]}
											<button
												onclick={() => clinicalPreviouslyBiopsied = val as any}
												class="flex-1 rounded-lg px-2 py-1.5 text-[10px] border transition-colors {clinicalPreviouslyBiopsied === val ? 'bg-teal-500/20 border-teal-500/40 text-teal-300' : 'bg-white/[0.03] border-white/[0.06] text-gray-400'}"
											>{label}</button>
										{/each}
									</div>
								</div>

								<div>
									<label class="mb-1 block text-[11px] text-gray-400">Family history of melanoma?</label>
									<div class="flex gap-2">
										{#each [["no", "No"], ["yes", "Yes"], ["unsure", "Not sure"]] as [val, label]}
											<button
												onclick={() => clinicalFamilyHistory = val as any}
												class="flex-1 rounded-lg px-2 py-1.5 text-[10px] border transition-colors {clinicalFamilyHistory === val ? 'bg-teal-500/20 border-teal-500/40 text-teal-300' : 'bg-white/[0.03] border-white/[0.06] text-gray-400'}"
											>{label}</button>
										{/each}
									</div>
								</div>

								<div>
									<label class="mb-1 block text-[11px] text-gray-400">Any symptoms?</label>
									<div class="flex gap-2 flex-wrap">
										{#each [["none", "None"], ["itching", "Itching"], ["bleeding", "Bleeding"], ["pain", "Pain"]] as [val, label]}
											<button
												onclick={() => {
													if (val === "none") { clinicalSymptoms = ["none"]; }
													else {
														const filtered = clinicalSymptoms.filter(s => s !== "none");
														if (filtered.includes(val as any)) { clinicalSymptoms = filtered.filter(s => s !== val); if (!clinicalSymptoms.length) clinicalSymptoms = ["none"]; }
														else { clinicalSymptoms = [...filtered, val as any]; }
													}
												}}
												class="rounded-lg px-3 py-1.5 text-[10px] border transition-colors {clinicalSymptoms.includes(val as any) ? 'bg-teal-500/20 border-teal-500/40 text-teal-300' : 'bg-white/[0.03] border-white/[0.06] text-gray-400'}"
											>{label}</button>
										{/each}
									</div>
								</div>
							</div>
						</div>
					{/if}
				</div>

				<!-- Captured — waiting for analysis to start (fallback if auto-analyze didn't fire) -->
			{:else}
				<div class="flex flex-col items-center gap-5 px-5 py-12 text-center animate-fadeIn">
					<div class="h-8 w-8 animate-spin rounded-full border-2 border-teal-500 border-t-transparent"></div>
					<p class="text-sm text-gray-400">Preparing analysis...</p>
					{#if capturedImageData && !analyzing}
						<button
							onclick={analyzeImage}
							class="mt-2 rounded-full bg-teal-600 px-8 py-3.5 text-[15px] font-semibold text-white hover:bg-teal-500 active:scale-95 transition-all touch-target"
						>
							Analyze Lesion
						</button>
					{/if}
				</div>
			{/if}
		</div>

	{:else if activeView === "history"}
		<div class="scrollbar-thin flex-1 overflow-y-auto px-5 pt-5 pb-24 overscroll-none">
			<LesionTimeline {records} />
		</div>

	{:else if activeView === "learn"}
		<AboutPage onclose={() => (activeView = "scan")} />

	{:else if activeView === "settings"}
		<div class="scrollbar-thin flex-1 overflow-y-auto px-5 pt-5 pb-24 overscroll-none">
			<div class="flex flex-col gap-5">
				<div class="card">
					<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Model</h3>
					<div class="flex items-center justify-between">
						<span class="text-[15px] text-gray-300">Version</span>
						<span class="text-sm font-mono text-gray-500">{modelVersion}</span>
					</div>
				</div>

				<div class="card">
					<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Analysis</h3>
					<label class="flex items-center justify-between">
						<div>
							<span class="text-[15px] text-gray-300">Multi-photo mode</span>
							<p class="text-[11px] text-gray-500 mt-0.5">Capture 2-3 photos for higher accuracy</p>
						</div>
						<input
							type="checkbox"
							bind:checked={multiImageMode}
							class="h-4 w-4 rounded border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
						/>
					</label>
				</div>

				<div class="card">
					<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Classification Mode</h3>
					<p class="text-[11px] text-gray-500 mb-3">Per-class thresholds from ADR-123 ROC analysis</p>
					<div class="flex flex-col gap-2">
						{#each [
							{ value: "screening", label: "Screening", desc: "Max cancer detection (fewer missed, more false alarms)" },
							{ value: "triage", label: "Balanced", desc: "Optimised sensitivity / specificity tradeoff" },
							{ value: "default", label: "Default", desc: "Standard argmax classification" },
						] as opt (opt.value)}
							<label
								class="flex items-start gap-3 rounded-xl px-3 py-2.5 transition-colors cursor-pointer
									{thresholdMode === opt.value
										? 'bg-teal-500/10 border border-teal-500/30'
										: 'border border-white/[0.04] hover:bg-white/[0.03]'}"
							>
								<input
									type="radio"
									name="thresholdMode"
									value={opt.value}
									checked={thresholdMode === opt.value}
									onchange={() => (thresholdMode = opt.value as ThresholdMode)}
									class="mt-0.5 h-4 w-4 border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
								/>
								<div class="min-w-0">
									<span class="text-[15px] text-gray-300">{opt.label}</span>
									<p class="text-[11px] text-gray-500 mt-0.5">{opt.desc}</p>
								</div>
							</label>
						{/each}
					</div>

					<!-- ADR-125: Ensemble Mode toggle -->
					<div class="mt-4 pt-4 border-t border-white/[0.06]">
						<label class="flex items-center justify-between">
							<div>
								<span class="text-[15px] text-gray-300">Ensemble Mode</span>
								<p class="text-[11px] text-gray-500 mt-0.5">Enable V1+V2 ensemble -- combines two models for higher accuracy</p>
								<p class="text-[10px] text-gray-600 mt-0.5">Requires both models to be available (doubles inference time)</p>
							</div>
							<input
								type="checkbox"
								bind:checked={ensembleEnabled}
								class="h-4 w-4 rounded border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
							/>
						</label>
					</div>
				</div>

				<div class="card">
					<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Inference</h3>
					<p class="text-[11px] text-gray-500 mb-3">Where classification runs (ADR-122)</p>
					<div class="flex flex-col gap-2">
						{#each [
							{ value: "auto", label: "Auto", desc: "Offline ONNX when loaded, otherwise HF API" },
							{ value: "online", label: "Online", desc: "Always use HF API ensemble (requires network)" },
							{ value: "offline", label: "Offline", desc: "ONNX model only (no data leaves device)" },
						] as opt (opt.value)}
							<label
								class="flex items-start gap-3 rounded-xl px-3 py-2.5 transition-colors cursor-pointer
									{inferenceStrategy === opt.value
										? 'bg-teal-500/10 border border-teal-500/30'
										: 'border border-white/[0.04] hover:bg-white/[0.03]'}"
							>
								<input
									type="radio"
									name="inferenceStrategy"
									value={opt.value}
									checked={inferenceStrategy === opt.value}
									onchange={() => {
										inferenceStrategy = opt.value as InferenceStrategy;
										if (opt.value !== "online" && !offlineModelReady) loadOfflineModel();
									}}
									class="mt-0.5 h-4 w-4 border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
								/>
								<div class="min-w-0">
									<span class="text-[15px] text-gray-300">{opt.label}</span>
									<p class="text-[11px] text-gray-500 mt-0.5">{opt.desc}</p>
								</div>
							</label>
						{/each}
					</div>
					<div class="mt-3 flex items-center gap-2">
						{#if offlineModelLoading}
							<span class="inline-block h-2 w-2 rounded-full bg-yellow-400 animate-pulse"></span>
							<span class="text-[11px] text-yellow-400">Loading ONNX model...</span>
						{:else if offlineModelReady}
							<span class="inline-block h-2 w-2 rounded-full bg-emerald-400"></span>
							<span class="text-[11px] text-emerald-400">Offline model ready</span>
						{:else}
							<span class="inline-block h-2 w-2 rounded-full bg-gray-600"></span>
							<span class="text-[11px] text-gray-500">Offline model not loaded</span>
						{/if}
					</div>
				</div>

				<div class="card">
					<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Brain Sync</h3>
					<div class="flex flex-col gap-3">
						<label class="flex items-center justify-between">
							<span class="text-[15px] text-gray-300">Enable sync</span>
							<input
								type="checkbox"
								bind:checked={brainSyncEnabled}
								class="h-4 w-4 rounded border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
							/>
						</label>
						<p class="text-[11px] text-gray-500">
							{brainSyncEnabled ? "Connected" : "Local-only mode"}
						</p>
					</div>
				</div>

				<div class="card">
					<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Network</h3>
					<div class="flex flex-col gap-3">
						<label class="flex items-center justify-between">
							<div>
								<span class="text-[15px] text-gray-300">Pi-brain collective intelligence</span>
								<p class="text-[11px] text-gray-500 mt-0.5">Share anonymized cases and retrieve similar cases from the network</p>
							</div>
							<input
								type="checkbox"
								bind:checked={networkEnabled}
								class="ml-3 h-4 w-4 flex-shrink-0 rounded border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
							/>
						</label>
						<p class="text-[11px] text-gray-500">
							{networkEnabled ? "Sharing and retrieval enabled -- no images leave your device" : "Disabled -- all data stays local"}
						</p>
					</div>
				</div>

				<div class="card">
					<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Privacy</h3>
					<div class="flex flex-col gap-3">
						<label class="flex items-center justify-between">
							<span class="text-[15px] text-gray-300">Strip EXIF data</span>
							<input
								type="checkbox"
								bind:checked={privacyStripExif}
								class="h-4 w-4 rounded border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
							/>
						</label>
						<label class="flex items-center justify-between">
							<span class="text-[15px] text-gray-300">Local processing only</span>
							<input
								type="checkbox"
								bind:checked={privacyLocalOnly}
								class="h-4 w-4 rounded border-white/[0.08] bg-white/[0.03] text-teal-500 focus:ring-teal-500/40"
							/>
						</label>
					</div>
				</div>

				<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] overflow-hidden">
					<h3 class="px-5 pt-5 pb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">Methodology</h3>
					<MethodologyPanel />
				</div>

				<!-- Analytics (moved from main nav to settings) -->
				<details class="rounded-2xl border border-white/[0.06] overflow-hidden">
					<summary class="px-5 py-4 text-[15px] font-medium text-gray-300 cursor-pointer hover:text-gray-200 transition-colors touch-target">
						Analytics Dashboard
					</summary>
					<div class="px-4 pb-4">
						<AnalyticsDashboard />
					</div>
				</details>

				<!-- Link to Learn page -->
				<button
					onclick={() => (activeView = "learn")}
					class="flex w-full items-center justify-between rounded-2xl border border-teal-500/15 bg-teal-500/5 p-5 text-left transition-all hover:bg-teal-500/10 active:scale-[0.98]"
				>
					<div class="flex items-center gap-3">
						<CarbonInformation class="h-5 w-5 text-teal-400" />
						<div>
							<p class="text-sm font-medium text-gray-200">How It Works</p>
							<p class="text-[11px] text-gray-500">Validation data, accuracy proof, limitations</p>
						</div>
					</div>
					<svg class="h-4 w-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
				</button>
			</div>
		</div>
	{/if}

	<!-- ===== BOTTOM NAVIGATION ===== -->
	<nav class="fixed bottom-0 left-0 right-0 z-40 flex border-t border-white/[0.04] bg-[#0a0a0f]/95 backdrop-blur-md" style="padding-bottom: env(safe-area-inset-bottom, 0px);">
		{#each NAV_ITEMS as item}
			<button
				onclick={() => (activeView = item.id)}
				class="flex flex-1 min-w-0 flex-col items-center justify-center gap-1 py-2.5 transition-colors {activeView === item.id ? 'text-teal-400 bg-teal-500/10' : 'text-gray-500 active:text-gray-400'}"
				style="min-height: 52px;"
				aria-label="{item.label} view"
				aria-current={activeView === item.id ? "page" : undefined}
			>
				<item.icon class="h-5 w-5 shrink-0" />
				<span class="text-[10px] font-medium leading-none tracking-wide">{item.label}</span>
			</button>
		{/each}
	</nav>
</div>

{#if showReferralLetter && classificationResult}
	<ReferralLetter
		onclose={() => (showReferralLetter = false)}
		classification={classificationResult}
		abcdeScores={abcdeScores}
		bodyLocation={capturedBodyLocation}
		patientAge={patientAge}
		patientSex={patientSex}
	/>
{/if}

<svelte:head>
	<meta name="theme-color" content="#0a0a0f" />
</svelte:head>