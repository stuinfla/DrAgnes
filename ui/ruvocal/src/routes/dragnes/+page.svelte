<script lang="ts">
	import CarbonArrowLeft from "~icons/carbon/arrow-left";
	import { base } from "$app/paths";
	import { onMount } from "svelte";

	let DrAgnesPanel: typeof import("$lib/components/dragnes/DrAgnesPanel.svelte").default | null =
		$state(null);
	let loadError: string | null = $state(null);
	let loading: boolean = $state(true);

	onMount(async () => {
		try {
			const mod = await import("$lib/components/dragnes/DrAgnesPanel.svelte");
			DrAgnesPanel = mod.default;
		} catch (err) {
			console.error("Failed to load DrAgnesPanel:", err);
			loadError = err instanceof Error ? err.message : "Failed to load DrAgnes components";
		} finally {
			loading = false;
		}
	});
</script>

<div class="flex h-dvh w-full flex-col bg-gray-50 dark:bg-gray-950">
	<!-- Header -->
	<header
		class="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-3 dark:border-gray-700 dark:bg-gray-900"
	>
		<a
			href="{base}/"
			class="flex h-9 w-9 items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
			aria-label="Back to chat"
		>
			<CarbonArrowLeft class="h-5 w-5" />
		</a>
		<div>
			<h1 class="text-base font-bold text-gray-900 dark:text-gray-100">DrAgnes</h1>
			<p class="text-xs text-gray-400">Dermatology Intelligence</p>
		</div>
	</header>

	<!-- Main panel -->
	<main class="flex-1 overflow-hidden">
		{#if loading}
			<div class="flex h-full items-center justify-center">
				<div class="flex flex-col items-center gap-3">
					<div
						class="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"
					></div>
					<p class="text-sm text-gray-400">Loading DrAgnes...</p>
				</div>
			</div>
		{:else if loadError}
			<div class="flex h-full items-center justify-center p-6">
				<div
					class="max-w-md rounded-xl border border-red-200 bg-red-50 p-6 text-center dark:border-red-800 dark:bg-red-950"
				>
					<h2 class="mb-2 text-lg font-semibold text-red-700 dark:text-red-400">
						Failed to load DrAgnes
					</h2>
					<p class="mb-4 text-sm text-red-600 dark:text-red-300">{loadError}</p>
					<button
						onclick={() => window.location.reload()}
						class="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
					>
						Retry
					</button>
				</div>
			</div>
		{:else if DrAgnesPanel}
			<DrAgnesPanel />
		{/if}
	</main>
</div>
