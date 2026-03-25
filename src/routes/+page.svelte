<script lang="ts">
	import DrAgnesPanel from '$lib/components/DrAgnesPanel.svelte';
</script>

<div class="flex h-dvh w-full flex-col bg-[#0a0a0f] overflow-hidden">
	<header class="flex shrink-0 items-center justify-between px-5 py-3 bg-[#0a0a0f]/95 backdrop-blur-md border-b border-white/[0.04]">
		<div class="flex items-center gap-2.5">
			<img src="/icons/icon-32x32.png" alt="Dr. Agnes" class="h-8 w-8 rounded-full object-cover ring-1 ring-white/[0.08]" />
			<div class="flex flex-col">
				<span class="text-[15px] font-semibold text-white tracking-tight leading-none">Dr. Agnes</span>
				<span class="text-[10px] text-teal-400 font-mono leading-none mt-0.5">v0.9.2</span>
			</div>
		</div>
		<span class="text-[10px] text-gray-500 uppercase tracking-widest font-medium">Research Preview</span>
	</header>
	<main class="flex-1 overflow-hidden">
		<DrAgnesPanel />
	</main>
</div>
