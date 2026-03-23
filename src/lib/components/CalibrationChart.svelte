<script lang="ts">
	import type { CalibrationBin } from "$lib/stores/analytics";

	interface Props {
		bins: CalibrationBin[];
		ece: number;
		hosmerP: number;
	}

	let { bins, ece, hosmerP }: Props = $props();

	const SVG_W = 300;
	const SVG_H = 220;
	const PAD = { top: 20, right: 20, bottom: 35, left: 45 };
	const plotW = SVG_W - PAD.left - PAD.right;
	const plotH = SVG_H - PAD.top - PAD.bottom;

	function x(v: number): number {
		return PAD.left + v * plotW;
	}
	function y(v: number): number {
		return PAD.top + (1 - v) * plotH;
	}

	let eceInterpretation = $derived(
		ece < 0.05 ? "Excellent calibration" :
		ece < 0.10 ? "Good calibration" :
		ece < 0.20 ? "Moderate miscalibration" :
		"Poor calibration -- predictions unreliable"
	);

	let eceColor = $derived(
		ece < 0.05 ? "text-teal-400" :
		ece < 0.10 ? "text-teal-400" :
		ece < 0.20 ? "text-amber-400" :
		"text-red-400"
	);

	let hlInterpretation = $derived(
		hosmerP > 0.10 ? "Good fit (p > 0.10)" :
		hosmerP > 0.05 ? "Acceptable fit (p > 0.05)" :
		"Poor fit -- model may be miscalibrated"
	);

	let populatedBins = $derived(bins.filter(b => b.count > 0));
	let populatedPolyline = $derived(
		populatedBins.map(b => `${x(b.predictedMean)},${y(b.observedFrequency)}`).join(" ")
	);
</script>

<div class="flex flex-col gap-4">
	<!-- ECE -->
	<div class="flex items-baseline gap-3">
		<span class="text-2xl font-bold {eceColor}">{(ece * 100).toFixed(1)}%</span>
		<span class="text-xs text-gray-400">ECE (Expected Calibration Error)</span>
	</div>
	<p class="text-xs {eceColor}">{eceInterpretation}</p>

	<!-- Calibration curve SVG -->
	<svg viewBox="0 0 {SVG_W} {SVG_H}" class="w-full max-w-sm" role="img" aria-label="Calibration curve showing predicted probability versus observed frequency across 10 bins">
		<!-- Background -->
		<rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="rgb(17,24,39)" rx="4" />

		<!-- Grid lines -->
		{#each [0, 0.25, 0.5, 0.75, 1.0] as tick}
			<line x1={x(0)} y1={y(tick)} x2={x(1)} y2={y(tick)} stroke="rgb(55,65,81)" stroke-width="0.5" />
			<line x1={x(tick)} y1={y(0)} x2={x(tick)} y2={y(1)} stroke="rgb(55,65,81)" stroke-width="0.5" />
			<text x={PAD.left - 5} y={y(tick) + 3} text-anchor="end" class="fill-gray-500" font-size="9">{(tick * 100).toFixed(0)}%</text>
			<text x={x(tick)} y={SVG_H - PAD.bottom + 15} text-anchor="middle" class="fill-gray-500" font-size="9">{(tick * 100).toFixed(0)}%</text>
		{/each}

		<!-- Perfect calibration diagonal -->
		<line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="rgb(107,114,128)" stroke-width="1" stroke-dasharray="4,3" />

		<!-- Calibration curve -->
		{#each bins as bin, i}
			{#if bin.count > 0}
				<circle
					cx={x(bin.predictedMean)}
					cy={y(bin.observedFrequency)}
					r={Math.max(3, Math.min(8, Math.sqrt(bin.count) * 1.5))}
					fill="rgb(20,184,166)"
					opacity="0.8"
				/>
			{/if}
		{/each}

		<!-- Connect points with a line -->
		{#if populatedBins.length > 1}
			<polyline
				points={populatedPolyline}
				fill="none"
				stroke="rgb(20,184,166)"
				stroke-width="1.5"
			/>
		{/if}

		<!-- Axis labels -->
		<text x={SVG_W / 2} y={SVG_H - 3} text-anchor="middle" class="fill-gray-400" font-size="10">Predicted Probability</text>
		<text x="12" y={SVG_H / 2} text-anchor="middle" class="fill-gray-400" font-size="10" transform="rotate(-90, 12, {SVG_H / 2})">Observed Frequency</text>
	</svg>

	<!-- Hosmer-Lemeshow -->
	<div class="flex items-baseline gap-2 text-xs">
		<span class="font-medium text-gray-300">Hosmer-Lemeshow p =</span>
		<span class="font-mono {hosmerP > 0.05 ? 'text-teal-400' : 'text-red-400'}">{hosmerP.toFixed(3)}</span>
		<span class="text-gray-500">{hlInterpretation}</span>
	</div>
</div>
