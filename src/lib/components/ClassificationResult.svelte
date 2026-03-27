<script lang="ts">
	import type {
		ClassificationResult,
		RiskLevel,
	} from "$lib/mela/types";
	import { LESION_LABELS } from "$lib/mela/types";
	import type { ABCDEScores } from "$lib/mela/types";
	import { getPrimaryICD10 } from "$lib/mela/icd10";

	interface Props {
		result: ClassificationResult;
		abcde?: ABCDEScores;
		onaction?: (action: string, payload?: unknown) => void;
		eventId?: string;
	}

	let { result, abcde, onaction, eventId }: Props = $props();

	const icd10 = $derived(getPrimaryICD10(result.topClass));

	let showModelProvenance: boolean = $state(false);

	type EnsembleWeight = { label: string; pct: number; color: string };
	const ensembleWeights: EnsembleWeight[] = $derived(
		result.usedDualModel
			? [{ label: "Dual ViT", pct: 50, color: "bg-teal-500" }, { label: "Trained", pct: 30, color: "bg-amber-500" }, { label: "Rules", pct: 20, color: "bg-orange-500" }]
			: result.usedHF
				? [{ label: "ViT", pct: 60, color: "bg-teal-500" }, { label: "Trained", pct: 25, color: "bg-amber-500" }, { label: "Rules", pct: 15, color: "bg-orange-500" }]
				: result.usedWasm
					? []
					: [{ label: "Trained", pct: 60, color: "bg-amber-500" }, { label: "Rules", pct: 40, color: "bg-orange-500" }]
	);
	// Feedback/pathology/correction functions removed for v1.0.0

	function confidenceColor(confidence: number): string {
		if (confidence >= 0.9) return "bg-green-500";
		if (confidence >= 0.7) return "bg-yellow-500";
		return "bg-red-500";
	}

	function riskColor(level: RiskLevel): string {
		const map: Record<RiskLevel, string> = {
			low: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
			moderate: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
			high: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
			critical: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
		};
		return map[level];
	}

	function riskIcon(level: RiskLevel): { svg: string; label: string } {
		const icons: Record<RiskLevel, { svg: string; label: string }> = {
			low: {
				svg: '<svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>',
				label: "LOW",
			},
			moderate: {
				svg: '<svg class="h-3 w-3" fill="currentColor" viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"></path></svg>',
				label: "MOD",
			},
			high: {
				svg: '<svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 9v2m0 4h.01M12 3a9 9 0 100 18 9 9 0 000-18z"></path></svg>',
				label: "HIGH",
			},
			critical: {
				svg: '<svg class="h-3 w-3" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" opacity="0"/><path d="M15.73 3H8.27L3 8.27v7.46L8.27 21h7.46L21 15.73V8.27L15.73 3zM12 17.3a1.3 1.3 0 110-2.6 1.3 1.3 0 010 2.6zm1-4.3h-2V7h2v6z"></path></svg>',
				label: "CRIT",
			},
		};
		return icons[level];
	}

	function pct(v: number): string {
		return `${(v * 100).toFixed(1)}%`;
	}

	function handleAction(action: string, payload?: unknown) {
		onaction?.(action, payload);
	}

	// correctTo removed for v1.0.0
</script>

<div class="flex flex-col gap-3 sm:gap-5 w-full">
	<!-- Top prediction -->
	<div class="rounded-xl border border-gray-200 bg-white p-3 sm:p-4 dark:border-gray-700 dark:bg-gray-800">
		<div class="mb-2 flex items-center justify-between">
			<h3 class="text-xs sm:text-sm font-semibold text-gray-500 dark:text-gray-400">Top Prediction</h3>
			{#if abcde}
				<span
					class="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold {riskColor(abcde.riskLevel)}"
				>
					{@html riskIcon(abcde.riskLevel).svg}
					{abcde.riskLevel.toUpperCase()}
				</span>
			{/if}
		</div>
		<p class="text-base sm:text-lg font-bold text-gray-900 dark:text-gray-100">
			<span class="font-mono">{result.topClass.toUpperCase()}</span>
			<span class="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">{LESION_LABELS[result.topClass]}</span>
			{#if icd10}
				<span class="ml-2 rounded bg-gray-800 px-1.5 py-0.5 text-[10px] font-mono text-gray-400">{icd10.code}</span>
			{/if}
		</p>
		<div class="mt-2 flex items-center gap-2">
			<div class="h-2.5 flex-1 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
				<div
					class="h-full rounded-full transition-all {confidenceColor(result.confidence)}"
					style="width: {result.confidence * 100}%"
				></div>
			</div>
			<span class="text-sm font-medium text-gray-600 dark:text-gray-300">
				{pct(result.confidence)}
			</span>
		</div>
		<p class="mt-1 text-xs text-gray-400">
			{result.inferenceTimeMs}ms &middot; {result.usedWasm ? "WASM" : result.usedHF ? "ViT + Local" : "Local"}
		</p>
	</div>

	<!-- All class probabilities -->
	<div class="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
		<h3 class="mb-3 text-sm font-semibold text-gray-500 dark:text-gray-400">
			Class Probabilities
		</h3>
		<div class="flex flex-col gap-2">
			{#each result.probabilities as prob}
				<div class="flex items-center gap-2">
					<span
						class="w-10 shrink-0 text-right text-xs font-mono text-gray-500 dark:text-gray-400"
					>
						{prob.className}
					</span>
					<div
						class="h-2 flex-1 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700"
					>
						<div
							class="h-full rounded-full bg-blue-500 dark:bg-blue-400 transition-all"
							style="width: {prob.probability * 100}%"
						></div>
					</div>
					<span class="w-12 text-right text-xs text-gray-500 dark:text-gray-400">
						{pct(prob.probability)}
					</span>
				</div>
			{/each}
		</div>
	</div>

	<!-- ABCDE Breakdown -->
	{#if abcde}
		<div
			class="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800"
		>
			<h3 class="mb-3 text-sm font-semibold text-gray-500 dark:text-gray-400">
				ABCDE Score Breakdown
			</h3>
			<p class="mt-1 mb-3 text-xs text-amber-400/70 italic">Scores estimated from classification confidence — not derived from direct image analysis</p>
			<div class="grid grid-cols-5 gap-1 sm:gap-2 text-center">
				{#each [
					{ key: "A", label: "Asymmetry", val: abcde.asymmetry, max: 2 },
					{ key: "B", label: "Border", val: abcde.border, max: 8 },
					{ key: "C", label: "Color", val: abcde.color, max: 6 },
					{ key: "D", label: "Diameter", val: abcde.diameterMm, max: 10 },
					{ key: "E", label: "Evolution", val: abcde.evolution, max: 2 },
				] as item}
					<div class="flex flex-col items-center gap-0.5 sm:gap-1">
						<span class="text-[10px] sm:text-xs text-gray-400">{item.key}</span>
						<div
							class="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-full border-2 text-xs sm:text-sm font-bold
								{item.val / item.max > 0.6
								? 'border-red-400 text-red-600 dark:text-red-400'
								: item.val / item.max > 0.3
									? 'border-yellow-400 text-yellow-600 dark:text-yellow-400'
									: 'border-green-400 text-green-600 dark:text-green-400'}"
						>
							{item.key === "D" ? item.val.toFixed(1) : item.val}
						</div>
						<span class="text-[10px] text-gray-400">{item.label}</span>
					</div>
				{/each}
			</div>
			<div class="mt-3 text-center">
				<span class="text-sm font-medium text-gray-600 dark:text-gray-300">
					Total Score: {abcde.totalScore.toFixed(1)}
				</span>
			</div>
		</div>
	{/if}

	<!-- Model Provenance -->
	<div class="rounded-xl border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
		<button
			onclick={() => (showModelProvenance = !showModelProvenance)}
			class="flex w-full items-center justify-between p-4 text-left"
			aria-expanded={showModelProvenance}
		>
			<h3 class="text-sm font-semibold text-gray-500 dark:text-gray-400 flex items-center gap-2">
				<svg class="h-4 w-4 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
				</svg>
				Model Provenance
			</h3>
			<svg
				class="h-4 w-4 text-gray-500 transition-transform {showModelProvenance ? 'rotate-180' : ''}"
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</button>
		{#if showModelProvenance}
			<div class="px-4 pb-4 space-y-2">
				<!-- Mode indicator -->
				<div class="rounded-lg bg-gray-800/50 p-2 text-[10px]">
					<p class="font-medium text-gray-300 mb-1">Classification Mode</p>
					{#if result.usedDualModel}
						<p class="text-teal-400">Dual-ViT Ensemble (online)</p>
						<p class="text-gray-500">Both neural networks contributed. Highest accuracy mode.</p>
					{:else if result.usedHF}
						<p class="text-teal-400">Single-ViT + Local Analysis (online)</p>
						<p class="text-gray-500">One neural network + local classifiers. One model may have been unavailable.</p>
					{:else if result.usedWasm}
						<p class="text-amber-400">WASM CNN (local)</p>
						<p class="text-gray-500">MobileNetV3 Small running locally via WebAssembly.</p>
					{:else}
						<p class="text-amber-400">Local Analysis Only (offline)</p>
						<p class="text-gray-500">No neural network available. Using literature-derived classifier + rule-based scoring.</p>
					{/if}
				</div>

				<!-- Models that contributed -->
				<div class="space-y-1.5">
					<p class="text-[10px] font-medium text-gray-400">Models Contributing to This Result</p>

					{#if result.usedDualModel || result.usedHF}
						<div class="rounded-lg bg-gray-800/50 p-2 text-[10px]">
							<div class="flex items-center justify-between">
								<p class="font-medium text-gray-300">Anwarkh1 ViT-Base</p>
								<span class="text-gray-500 font-mono">85.8M params</span>
							</div>
							<p class="text-gray-500 mt-0.5">Fine-tuned on HAM10000 + ISIC 2019 combined (37,484 dermoscopy images, 7 classes). HuggingFace-hosted.</p>
						</div>
					{/if}

					{#if result.usedDualModel}
						<div class="rounded-lg bg-gray-800/50 p-2 text-[10px]">
							<div class="flex items-center justify-between">
								<p class="font-medium text-gray-300">SigLIP (SkinTag Labs)</p>
								<span class="text-gray-500 font-mono">400M params</span>
							</div>
							<p class="text-gray-500 mt-0.5">SigLIP vision-language model trained on dermoscopy image-text pairs. HuggingFace-hosted.</p>
						</div>
					{/if}

					{#if result.usedWasm}
						<div class="rounded-lg bg-gray-800/50 p-2 text-[10px]">
							<div class="flex items-center justify-between">
								<p class="font-medium text-gray-300">MobileNetV3 Small (WASM)</p>
								<span class="text-gray-500 font-mono">~2.5M params</span>
							</div>
							<p class="text-gray-500 mt-0.5">Lightweight CNN running locally via @ruvector/cnn WebAssembly module.</p>
						</div>
					{/if}

					<div class="rounded-lg bg-gray-800/50 p-2 text-[10px]">
						<div class="flex items-center justify-between">
							<p class="font-medium text-gray-300">Literature-Derived Classifier</p>
							<span class="text-gray-500 font-mono">20 features</span>
						</div>
						<p class="text-gray-500 mt-0.5">Logistic regression with weights from Stolz 1994, Argenziano 1998, DermNet NZ. Runs locally.</p>
					</div>

					<div class="rounded-lg bg-gray-800/50 p-2 text-[10px]">
						<div class="flex items-center justify-between">
							<p class="font-medium text-gray-300">Rule-Based Clinical Scoring</p>
							<span class="text-gray-500 font-mono">TDS + 7-point</span>
						</div>
						<p class="text-gray-500 mt-0.5">TDS formula (Stolz ABCD) + 7-point checklist (Argenziano) + melanoma safety gate. Runs locally.</p>
					</div>
				</div>

				<!-- Ensemble weights -->
				<div class="rounded-lg bg-gray-800/50 p-2 text-[10px]">
					<p class="font-medium text-gray-400 mb-1.5">Ensemble Weights Used</p>
					{#if ensembleWeights.length > 0}
						<div class="space-y-1">
							{#each ensembleWeights as w}
								<div class="flex items-center gap-2">
									<div class="h-1.5 flex-1 rounded-full bg-gray-700 overflow-hidden">
										<div class="h-full rounded-full {w.color}" style="width: {w.pct}%"></div>
									</div>
									<span class="text-gray-400 w-20 text-right">{w.label} {w.pct}%</span>
								</div>
							{/each}
						</div>
					{:else}
						<p class="text-gray-500">MobileNetV3 WASM model -- single model, no ensemble.</p>
					{/if}
				</div>
				<p class="text-[10px] font-mono text-gray-600">Model ID: {result.modelId}</p>
			</div>
		{/if}
	</div>

	<!-- Action buttons — only Dismiss kept for v1.0.0 -->
	<div class="sticky bottom-0 bg-gray-950/95 backdrop-blur-sm border-t border-gray-800 p-3 -mx-3 sm:-mx-4 mt-4">
		<div class="text-center">
			<button
				onclick={() => handleAction("dismiss")}
				class="text-sm text-gray-400 underline hover:text-gray-300"
			>
				Dismiss
			</button>
		</div>
	</div>
</div>
