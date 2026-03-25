<script lang="ts">
	import type { DiscordanceBreakdown, DiscordantCase } from "$lib/stores/analytics";
	import { LESION_LABELS } from "$lib/mela/types";
	import type { LesionClass } from "$lib/mela/types";

	interface Props {
		breakdown: DiscordanceBreakdown;
		cases: DiscordantCase[];
	}

	let { breakdown, cases }: Props = $props();

	const COLORS: Record<string, string> = {
		overcalled: "#f59e0b",
		missed: "#ef4444",
		artifact: "#6b7280",
		edge_case: "#8b5cf6",
		other: "#3b82f6",
	};

	const LABELS: Record<string, string> = {
		overcalled: "Overcalled",
		missed: "Missed",
		artifact: "Artifact",
		edge_case: "Edge Case",
		other: "Other",
	};

	let total = $derived(
		breakdown.overcalled + breakdown.missed + breakdown.artifact + breakdown.edge_case + breakdown.other
	);

	let segments = $derived.by(() => {
		if (total === 0) return [];
		const entries = Object.entries(breakdown).filter(([, v]) => v > 0);
		const segs: { key: string; count: number; pct: number; color: string; startAngle: number; endAngle: number }[] = [];
		let cumAngle = 0;
		for (const [key, count] of entries) {
			const pct = count / total;
			const angle = pct * 2 * Math.PI;
			segs.push({
				key,
				count,
				pct,
				color: COLORS[key] ?? "#6b7280",
				startAngle: cumAngle,
				endAngle: cumAngle + angle,
			});
			cumAngle += angle;
		}
		return segs;
	});

	function arcPath(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
		const start = {
			x: cx + r * Math.cos(startAngle - Math.PI / 2),
			y: cy + r * Math.sin(startAngle - Math.PI / 2),
		};
		const end = {
			x: cx + r * Math.cos(endAngle - Math.PI / 2),
			y: cy + r * Math.sin(endAngle - Math.PI / 2),
		};
		const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
		return `M ${cx} ${cy} L ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y} Z`;
	}

	function labelForClass(cls: string): string {
		return (LESION_LABELS as Record<string, string>)[cls] ?? cls;
	}
</script>

<div class="flex flex-col gap-4">
	{#if total === 0}
		<p class="text-xs text-gray-500">No discordant cases recorded</p>
	{:else}
		<div class="flex items-start gap-6">
			<!-- Pie chart -->
			<svg viewBox="0 0 120 120" class="w-28 h-28 flex-shrink-0" role="img" aria-label="Pie chart showing discordance categories">
				{#each segments as seg}
					<path d={arcPath(60, 60, 50, seg.startAngle, seg.endAngle)} fill={seg.color} />
				{/each}
				<circle cx="60" cy="60" r="22" fill="rgb(3,7,18)" />
				<text x="60" y="63" text-anchor="middle" class="fill-gray-300" font-size="14" font-weight="bold">{total}</text>
			</svg>

			<!-- Legend -->
			<div class="flex flex-col gap-1.5 text-xs">
				{#each segments as seg}
					<div class="flex items-center gap-2">
						<span class="inline-block h-2.5 w-2.5 rounded-sm flex-shrink-0" style="background-color: {seg.color}"></span>
						<span class="text-gray-300">{LABELS[seg.key] ?? seg.key}</span>
						<span class="text-gray-500">{seg.count} ({(seg.pct * 100).toFixed(0)}%)</span>
					</div>
				{/each}
			</div>
		</div>

		<!-- Recent discordant cases -->
		{#if cases.length > 0}
			<div class="mt-2">
				<h4 class="text-xs font-medium text-gray-400 mb-2">Recent Discordant Cases</h4>
				<div class="flex flex-col gap-1 max-h-40 overflow-y-auto scrollbar-custom">
					{#each cases.slice(0, 10) as c}
						<div class="flex items-center gap-2 rounded-lg bg-gray-800/50 px-3 py-1.5 text-xs">
							<span class="inline-block h-2 w-2 rounded-full flex-shrink-0" style="background-color: {COLORS[c.reason] ?? '#6b7280'}"></span>
							<span class="text-gray-300 truncate">{labelForClass(c.predictedClass)}</span>
							{#if c.pathologyResult}
								<span class="text-gray-500">-></span>
								<span class="text-gray-300 truncate">{labelForClass(c.pathologyResult)}</span>
							{/if}
							<span class="ml-auto text-gray-500 flex-shrink-0">{c.timestamp.slice(0, 10)}</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}
</div>
