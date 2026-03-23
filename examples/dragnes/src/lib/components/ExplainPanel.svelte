<script lang="ts">
	interface Finding {
		feature: string;
		value: string;
		impact: "supports" | "opposes" | "neutral";
		weight: "strong" | "moderate" | "weak";
		citation: string;
		clinicalSignificance: string;
		tdsWeight: string;
	}

	interface Props {
		topClass: string;
		findings: Finding[];
	}

	let { topClass, findings }: Props = $props();

	/** Track which findings are expanded by index */
	let expandedSet: Set<number> = $state(new Set());

	function toggleDetail(index: number) {
		const next = new Set(expandedSet);
		if (next.has(index)) {
			next.delete(index);
		} else {
			next.add(index);
		}
		expandedSet = next;
	}

	function expandAll() {
		expandedSet = new Set(findings.map((_, i) => i));
	}

	function collapseAll() {
		expandedSet = new Set();
	}

	function weightBadgeColor(w: Finding["weight"]): string {
		if (w === "strong") return "bg-red-500/20 text-red-400 border-red-500/30";
		if (w === "moderate") return "bg-amber-500/20 text-amber-400 border-amber-500/30";
		return "bg-gray-500/20 text-gray-400 border-gray-500/30";
	}
</script>

<div class="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
	<div class="mb-3 flex items-center justify-between">
		<h3 class="text-sm font-semibold text-gray-200 flex items-center gap-2">
			<svg class="h-4 w-4 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
			</svg>
			Why "{topClass}"?
		</h3>
		{#if findings.length > 0}
			<div class="flex gap-2">
				<button
					onclick={expandAll}
					class="text-[10px] text-teal-400 hover:text-teal-300 transition-colors py-1 px-1"
				>
					Expand all
				</button>
				<span class="text-gray-600">|</span>
				<button
					onclick={collapseAll}
					class="text-[10px] text-gray-500 hover:text-gray-400 transition-colors py-1 px-1"
				>
					Collapse all
				</button>
			</div>
		{/if}
	</div>
	{#if findings.length === 0}
		<p class="text-xs text-gray-500 italic">No detailed feature analysis available for this classification.</p>
	{:else}
		<div class="space-y-2">
			{#each findings as f, i}
				{@const isExpanded = expandedSet.has(i)}
				<div class="rounded-lg bg-gray-800/40 transition-colors {isExpanded ? 'bg-gray-800/60' : ''}">
					<button
						onclick={() => toggleDetail(i)}
						class="flex w-full items-start gap-2 p-2.5 sm:p-2.5 text-left text-xs touch-target"
						style="min-height: 44px;"
						aria-expanded={isExpanded}
						aria-controls="finding-detail-{i}"
					>
						<span class="mt-0.5 flex-shrink-0">
							{#if f.impact === "supports"}
								<span class="text-red-400">&#9650;</span>
							{:else if f.impact === "opposes"}
								<span class="text-green-400">&#9660;</span>
							{:else}
								<span class="text-gray-500">&#9679;</span>
							{/if}
						</span>
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2 flex-wrap">
								<span class="text-gray-300"><strong>{f.feature}:</strong> {f.value}</span>
								<span class="inline-flex rounded-full border px-1.5 py-0 text-[9px] font-medium {weightBadgeColor(f.weight)}">
									{f.weight}
								</span>
							</div>
							<span class="text-gray-500 italic text-[10px]">{f.citation}</span>
						</div>
						<svg
							class="h-3.5 w-3.5 flex-shrink-0 text-gray-500 transition-transform mt-0.5 {isExpanded ? 'rotate-180' : ''}"
							fill="none" stroke="currentColor" viewBox="0 0 24 24"
						>
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
						</svg>
					</button>
					{#if isExpanded}
						<div
							id="finding-detail-{i}"
							class="px-3 sm:px-2.5 pb-3 sm:pb-2.5 pt-0 ml-4 sm:ml-6 border-t border-gray-700/50"
						>
							<div class="mt-2 space-y-1.5">
								<div>
									<p class="text-[10px] font-medium text-teal-400">Clinical significance</p>
									<p class="text-[10px] text-gray-400">{f.clinicalSignificance}</p>
								</div>
								<div>
									<p class="text-[10px] font-medium text-amber-400">Classification weight</p>
									<p class="text-[10px] text-gray-400">{f.tdsWeight}</p>
								</div>
							</div>
						</div>
					{/if}
				</div>
			{/each}
		</div>
		<div class="mt-3 flex gap-3 sm:gap-4 text-[9px] sm:text-[10px] text-gray-500 flex-wrap">
			<span class="flex items-center gap-1"><span class="text-red-400">&#9650;</span> Supports</span>
			<span class="flex items-center gap-1"><span class="text-green-400">&#9660;</span> Opposes</span>
			<span class="flex items-center gap-1"><span class="text-gray-500">&#9679;</span> Neutral</span>
		</div>
		<p class="mt-2 text-[10px] text-gray-600 italic">
			Click any finding to see its clinical significance and how much it influenced the classification.
		</p>
	{/if}
</div>
