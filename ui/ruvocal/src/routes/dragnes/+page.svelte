<script lang="ts">
	import DrAgnesPanel from "$lib/components/dragnes/DrAgnesPanel.svelte";
	import CarbonArrowLeft from "~icons/carbon/arrow-left";
	import { base } from "$app/paths";
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
			<h1 class="text-base font-bold text-gray-900 dark:text-gray-100">
				DrAgnes
			</h1>
			<p class="text-xs text-gray-400">Dermatology Intelligence</p>
		</div>
	</header>

	<!-- Main panel -->
	<main class="flex-1 overflow-hidden">
		<DrAgnesPanel />
	</main>
</div>
