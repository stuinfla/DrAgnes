<script lang="ts">
	import type { RollingConcordance } from "$lib/stores/analytics";

	interface Props {
		data: RollingConcordance[];
	}

	let { data }: Props = $props();

	const SVG_W = 340;
	const SVG_H = 180;
	const PAD = { top: 15, right: 15, bottom: 35, left: 45 };
	const plotW = SVG_W - PAD.left - PAD.right;
	const plotH = SVG_H - PAD.top - PAD.bottom;

	let hasData = $derived(data.length >= 2);

	function x(i: number, total: number): number {
		if (total <= 1) return PAD.left + plotW / 2;
		return PAD.left + (i / (total - 1)) * plotW;
	}
	function y(v: number): number {
		return PAD.top + (1 - v) * plotH;
	}

	let polylinePoints = $derived(
		data.map((d, i) => `${x(i, data.length)},${y(d.concordanceRate)}`).join(" ")
	);

	// Determine if trend is improving: compare first-half mean vs second-half mean
	let trendLabel = $derived.by(() => {
		if (data.length < 4) return "Insufficient data for trend";
		const mid = Math.floor(data.length / 2);
		const firstHalf = data.slice(0, mid).reduce((s, d) => s + d.concordanceRate, 0) / mid;
		const secondHalf = data.slice(mid).reduce((s, d) => s + d.concordanceRate, 0) / (data.length - mid);
		const diff = secondHalf - firstHalf;
		if (diff > 0.02) return "Improving";
		if (diff < -0.02) return "Declining";
		return "Stable";
	});

	let trendColor = $derived(
		trendLabel === "Improving" ? "text-teal-400" :
		trendLabel === "Declining" ? "text-red-400" :
		"text-gray-400"
	);

	// Compute x-axis date labels (show first, middle, last)
	let dateLabels = $derived.by(() => {
		if (data.length === 0) return [];
		const labels: { x: number; label: string }[] = [];
		const indices = data.length === 1 ? [0] :
			data.length === 2 ? [0, data.length - 1] :
			[0, Math.floor(data.length / 2), data.length - 1];
		for (const i of indices) {
			labels.push({
				x: x(i, data.length),
				label: data[i].date.slice(5), // MM-DD
			});
		}
		return labels;
	});
</script>

<div class="flex flex-col gap-3">
	{#if !hasData}
		<p class="text-xs text-gray-500">Need at least 2 data points for trend chart</p>
	{:else}
		<svg viewBox="0 0 {SVG_W} {SVG_H}" class="w-full" role="img" aria-label="30-day rolling concordance trend chart">
			<rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="rgb(17,24,39)" rx="4" />

			<!-- Y grid -->
			{#each [0, 0.25, 0.5, 0.75, 1.0] as tick}
				<line x1={PAD.left} y1={y(tick)} x2={PAD.left + plotW} y2={y(tick)} stroke="rgb(55,65,81)" stroke-width="0.5" />
				<text x={PAD.left - 5} y={y(tick) + 3} text-anchor="end" class="fill-gray-500" font-size="9">{(tick * 100).toFixed(0)}%</text>
			{/each}

			<!-- Trend line -->
			<polyline
				points={polylinePoints}
				fill="none"
				stroke="rgb(20,184,166)"
				stroke-width="2"
				stroke-linejoin="round"
			/>

			<!-- Data points -->
			{#each data as d, i}
				<circle cx={x(i, data.length)} cy={y(d.concordanceRate)} r="2.5" fill="rgb(20,184,166)" />
			{/each}

			<!-- X labels -->
			{#each dateLabels as dl}
				<text x={dl.x} y={SVG_H - PAD.bottom + 15} text-anchor="middle" class="fill-gray-500" font-size="9">{dl.label}</text>
			{/each}

			<!-- Axis labels -->
			<text x={SVG_W / 2} y={SVG_H - 3} text-anchor="middle" class="fill-gray-400" font-size="10">Date</text>
			<text x="12" y={SVG_H / 2} text-anchor="middle" class="fill-gray-400" font-size="10" transform="rotate(-90, 12, {SVG_H / 2})">Concordance</text>
		</svg>
	{/if}

	<!-- Learning curve label -->
	<div class="flex items-center gap-2">
		<span class="text-xs text-gray-400">Learning Curve:</span>
		<span class="text-xs font-semibold {trendColor}">{trendLabel}</span>
	</div>
</div>
