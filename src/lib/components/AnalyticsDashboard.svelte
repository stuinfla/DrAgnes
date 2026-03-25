<script lang="ts">
	import { computeMetrics, clearAnalytics, type PracticeMetrics } from "$lib/stores/analytics";
	import { LESION_LABELS } from "$lib/mela/types";
	import type { LesionClass } from "$lib/mela/types";
	import { DERMASENSOR_BENCHMARKS, MELA_TARGETS } from "$lib/mela/clinical-baselines";

	import CalibrationChart from "./CalibrationChart.svelte";
	import TrendChart from "./TrendChart.svelte";
	import DiscordancePieChart from "./DiscordancePieChart.svelte";

	// Recompute metrics reactively (on mount and when tab becomes visible)
	let metrics: PracticeMetrics = $state(computeMetrics());
	let showClearConfirm: boolean = $state(false);

	function refresh() {
		metrics = computeMetrics();
	}

	function handleClear() {
		clearAnalytics();
		metrics = computeMetrics();
		showClearConfirm = false;
	}

	function pct(v: number): string {
		return (v * 100).toFixed(1) + "%";
	}

	function pctCI(ci: [number, number]): string {
		return `[${(ci[0] * 100).toFixed(0)}-${(ci[1] * 100).toFixed(0)}]`;
	}

	function insufficientData(n: number): boolean {
		return n < 10;
	}

	function labelForClass(cls: string): string {
		return (LESION_LABELS as Record<string, string>)[cls] ?? cls;
	}

	// Fitzpatrick disparity detection
	let maxFitzAccuracy = $derived(Math.max(...metrics.fitzpatrick.filter(r => r.n > 0).map(r => r.accuracy), 0));
</script>

<div class="flex flex-col gap-6 pb-16">
	<!-- Header + refresh -->
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-lg font-bold text-gray-100">Your Practice Performance</h2>
			<p class="text-xs text-gray-500">Computed from your outcome feedback</p>
		</div>
		<button
			onclick={refresh}
			class="rounded-lg border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700 transition-colors"
		>
			Refresh
		</button>
	</div>

	<!-- Section A: Key Metrics Grid -->
	<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
		<!-- Concordance Rate -->
		<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
			<p class="text-xs text-gray-500 mb-1">Concordance Rate</p>
			{#if insufficientData(metrics.concordanceN)}
				<p class="text-xl font-bold text-gray-600">--</p>
				<p class="text-[10px] text-gray-600">Insufficient data</p>
			{:else}
				<p class="text-2xl font-bold {metrics.concordanceRate >= 0.85 ? 'text-teal-400' : metrics.concordanceRate >= 0.70 ? 'text-amber-400' : 'text-red-400'}">
					{pct(metrics.concordanceRate)}
				</p>
			{/if}
			<p class="text-[10px] text-gray-500 mt-1">n={metrics.concordanceN}</p>
		</div>

		<!-- NNB -->
		<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
			<p class="text-xs text-gray-500 mb-1">Biopsy-to-Malignancy</p>
			{#if insufficientData(metrics.nnbN)}
				<p class="text-xl font-bold text-gray-600">--</p>
				<p class="text-[10px] text-gray-600">Insufficient data</p>
			{:else}
				<p class="text-2xl font-bold {metrics.nnb > 0 && metrics.nnb < 4.5 ? 'text-teal-400' : metrics.nnb <= 6 ? 'text-amber-400' : 'text-red-400'}">
					{metrics.nnb > 0 ? metrics.nnb.toFixed(1) : "--"}
				</p>
				<p class="text-[10px] text-gray-500">NNB target &lt; 4.5</p>
			{/if}
			<p class="text-[10px] text-gray-500 mt-1">n={metrics.nnbN}</p>
		</div>

		<!-- Scans Analyzed -->
		<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
			<p class="text-xs text-gray-500 mb-1">Scans Analyzed</p>
			<p class="text-2xl font-bold text-gray-200">{metrics.totalScans}</p>
			<p class="text-[10px] text-gray-500 mt-1">total</p>
		</div>

		<!-- Feedback Rate -->
		<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
			<p class="text-xs text-gray-500 mb-1">Feedback Rate</p>
			{#if metrics.totalScans === 0}
				<p class="text-xl font-bold text-gray-600">--</p>
			{:else}
				<p class="text-2xl font-bold {metrics.feedbackRate >= 0.8 ? 'text-teal-400' : metrics.feedbackRate >= 0.5 ? 'text-amber-400' : 'text-red-400'}">
					{pct(metrics.feedbackRate)}
				</p>
			{/if}
			<p class="text-[10px] text-gray-500 mt-1">n={metrics.feedbackN} / {metrics.totalScans}</p>
		</div>
	</div>

	<!-- Section B: Per-Class Accuracy Table -->
	<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
		<h3 class="text-sm font-semibold text-gray-400 mb-3">Per-Class Accuracy</h3>
		{#if metrics.perClass.every(c => c.n === 0)}
			<p class="text-xs text-gray-600">No pathology outcomes recorded yet. Record pathology results to see per-class accuracy.</p>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-xs">
					<thead>
						<tr class="border-b border-gray-800 text-gray-500">
							<th class="text-left py-2 pr-3 font-medium">Class</th>
							<th class="text-right py-2 px-2 font-medium">Sens.</th>
							<th class="text-right py-2 px-2 font-medium">Spec.</th>
							<th class="text-right py-2 px-2 font-medium">PPV</th>
							<th class="text-right py-2 px-2 font-medium">NPV</th>
							<th class="text-right py-2 px-2 font-medium">95% CI</th>
							<th class="text-right py-2 pl-2 font-medium">N</th>
						</tr>
					</thead>
					<tbody>
						{#each metrics.perClass as row}
							<tr class="border-b border-gray-800/50 {row.n === 0 ? 'opacity-40' : ''}">
								<td class="py-1.5 pr-3">
									<span class="font-mono text-gray-300">{row.className}</span>
									<span class="ml-1 text-gray-500 hidden sm:inline">{labelForClass(row.className)}</span>
								</td>
								<td class="text-right py-1.5 px-2 text-gray-300">
									{#if insufficientData(row.n)}
										<span class="text-gray-600">--</span>
									{:else}
										{pct(row.sensitivity)}
									{/if}
								</td>
								<td class="text-right py-1.5 px-2 text-gray-300">
									{#if insufficientData(row.n)}
										<span class="text-gray-600">--</span>
									{:else}
										{pct(row.specificity)}
									{/if}
								</td>
								<td class="text-right py-1.5 px-2 text-gray-300">
									{#if insufficientData(row.n)}
										<span class="text-gray-600">--</span>
									{:else}
										{pct(row.ppv)}
									{/if}
								</td>
								<td class="text-right py-1.5 px-2 text-gray-300">
									{#if insufficientData(row.n)}
										<span class="text-gray-600">--</span>
									{:else}
										{pct(row.npv)}
									{/if}
								</td>
								<td class="text-right py-1.5 px-2 font-mono text-gray-400">
									{#if insufficientData(row.n)}
										<span class="text-gray-600">--</span>
									{:else}
										{pctCI(row.ci95)}
									{/if}
								</td>
								<td class="text-right py-1.5 pl-2 text-gray-500">{row.n}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>

	<!-- Section C: Calibration -->
	<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
		<h3 class="text-sm font-semibold text-gray-400 mb-3">Calibration</h3>
		{#if metrics.totalScans === 0}
			<p class="text-xs text-gray-600">No data. Analyze lesions and record pathology outcomes to see calibration metrics.</p>
		{:else}
			<CalibrationChart
				bins={metrics.calibrationBins}
				ece={metrics.ece}
				hosmerP={metrics.hosmerLemeshowP}
			/>
		{/if}
	</div>

	<!-- Section D: Fitzpatrick Equity -->
	<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
		<h3 class="text-sm font-semibold text-gray-400 mb-3">Fitzpatrick Equity</h3>
		{#if metrics.fitzpatrick.every(r => r.n === 0)}
			<p class="text-xs text-gray-600">No Fitzpatrick skin type data recorded. Add Fitzpatrick type in demographics during capture.</p>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-xs">
					<thead>
						<tr class="border-b border-gray-800 text-gray-500">
							<th class="text-left py-2 pr-3 font-medium">Skin Type</th>
							<th class="text-right py-2 px-2 font-medium">Accuracy</th>
							<th class="text-right py-2 pl-2 font-medium">N</th>
						</tr>
					</thead>
					<tbody>
						{#each metrics.fitzpatrick as row}
							{@const disparity = row.n > 0 && maxFitzAccuracy > 0 ? maxFitzAccuracy - row.accuracy : 0}
							<tr class="border-b border-gray-800/50 {row.n === 0 ? 'opacity-40' : ''}">
								<td class="py-1.5 pr-3 text-gray-300">Type {row.type}</td>
								<td class="text-right py-1.5 px-2 {disparity > 0.05 ? 'text-red-400 font-semibold' : 'text-gray-300'}">
									{#if row.n === 0}
										<span class="text-gray-600">--</span>
									{:else if insufficientData(row.n)}
										<span class="text-gray-500">{pct(row.accuracy)}</span>
										<span class="text-gray-600 text-[10px] ml-1">(low n)</span>
									{:else}
										{pct(row.accuracy)}
										{#if disparity > 0.05}
											<span class="text-[10px] text-red-500 ml-1">-{(disparity * 100).toFixed(1)}pp</span>
										{/if}
									{/if}
								</td>
								<td class="text-right py-1.5 pl-2 text-gray-500">{row.n}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>

	<!-- Section E: Discordance Analysis -->
	<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
		<h3 class="text-sm font-semibold text-gray-400 mb-3">Discordance Analysis</h3>
		<DiscordancePieChart
			breakdown={metrics.discordance}
			cases={metrics.discordantCases}
		/>
	</div>

	<!-- Section F: Trends -->
	<div class="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
		<h3 class="text-sm font-semibold text-gray-400 mb-3">30-Day Rolling Concordance</h3>
		{#if metrics.rolling30Day.length === 0}
			<p class="text-xs text-gray-600">Record feedback on scans to see concordance trends over time.</p>
		{:else}
			<TrendChart data={metrics.rolling30Day} />
		{/if}
	</div>

	<!-- Section G: Global Validation (separate section) -->
	<div class="rounded-xl border border-dashed border-gray-700 bg-gray-950/60 p-4">
		<div class="flex items-center gap-2 mb-3">
			<h3 class="text-sm font-semibold text-gray-400">Global Validation Benchmarks</h3>
			<span class="rounded-full border border-gray-700 bg-gray-800 px-2 py-0.5 text-[10px] text-gray-500 uppercase tracking-wide">Curated dataset -- not your practice data</span>
		</div>

		<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
			<!-- DermaSensor FDA benchmark -->
			<div class="flex flex-col gap-2">
				<h4 class="text-xs font-medium text-gray-500">DermaSensor (FDA DEN230008)</h4>
				<div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
					<span class="text-gray-500">Melanoma Sens.</span>
					<span class="text-gray-300 font-mono">{(DERMASENSOR_BENCHMARKS.sensitivity.melanoma * 100).toFixed(1)}%</span>
					<span class="text-gray-500">BCC Sens.</span>
					<span class="text-gray-300 font-mono">{(DERMASENSOR_BENCHMARKS.sensitivity.bcc * 100).toFixed(1)}%</span>
					<span class="text-gray-500">Overall Spec.</span>
					<span class="text-gray-300 font-mono">{(DERMASENSOR_BENCHMARKS.specificity.overall * 100).toFixed(1)}%</span>
					<span class="text-gray-500">NPV</span>
					<span class="text-gray-300 font-mono">{(DERMASENSOR_BENCHMARKS.npv * 100).toFixed(1)}%</span>
					<span class="text-gray-500">AUROC</span>
					<span class="text-gray-300 font-mono">{DERMASENSOR_BENCHMARKS.auroc.toFixed(3)}</span>
					<span class="text-gray-500">FST I-III Sens.</span>
					<span class="text-gray-300 font-mono">{(DERMASENSOR_BENCHMARKS.fitzpatrick.fst_1_3.sensitivity * 100).toFixed(0)}%</span>
					<span class="text-gray-500">FST IV-VI Sens.</span>
					<span class="text-gray-300 font-mono">{(DERMASENSOR_BENCHMARKS.fitzpatrick.fst_4_6.sensitivity * 100).toFixed(0)}%</span>
				</div>
			</div>

			<!-- Mela targets -->
			<div class="flex flex-col gap-2">
				<h4 class="text-xs font-medium text-gray-500">Mela Targets</h4>
				<div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
					<span class="text-gray-500">Melanoma Sens.</span>
					<span class="text-teal-400 font-mono">{(MELA_TARGETS.sensitivity.melanoma * 100).toFixed(0)}%</span>
					<span class="text-gray-500">BCC Sens.</span>
					<span class="text-teal-400 font-mono">{(MELA_TARGETS.sensitivity.bcc * 100).toFixed(0)}%</span>
					<span class="text-gray-500">Spec. (target)</span>
					<span class="text-teal-400 font-mono">{(MELA_TARGETS.specificity.target * 100).toFixed(0)}%</span>
					<span class="text-gray-500">NPV (min)</span>
					<span class="text-teal-400 font-mono">{(MELA_TARGETS.npv.minimum * 100).toFixed(0)}%</span>
					<span class="text-gray-500">FNR ceiling (mel)</span>
					<span class="text-teal-400 font-mono">{(MELA_TARGETS.falseNegativeRate.melanomaCeiling * 100).toFixed(0)}%</span>
					<span class="text-gray-500">FST max gap</span>
					<span class="text-teal-400 font-mono">{(MELA_TARGETS.fitzpatrickDisparity.maxGap * 100).toFixed(0)}%</span>
				</div>
			</div>
		</div>
	</div>

	<!-- Clear data -->
	<div class="flex justify-center">
		{#if showClearConfirm}
			<div class="flex items-center gap-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2">
				<span class="text-xs text-red-400">Clear all analytics data? This cannot be undone.</span>
				<button
					onclick={handleClear}
					class="rounded bg-red-600 px-3 py-1 text-xs font-medium text-white hover:bg-red-700"
				>
					Confirm
				</button>
				<button
					onclick={() => (showClearConfirm = false)}
					class="rounded bg-gray-700 px-3 py-1 text-xs text-gray-300 hover:bg-gray-600"
				>
					Cancel
				</button>
			</div>
		{:else}
			<button
				onclick={() => (showClearConfirm = true)}
				class="text-xs text-gray-600 hover:text-gray-400 transition-colors"
			>
				Clear All Analytics Data
			</button>
		{/if}
	</div>
</div>
