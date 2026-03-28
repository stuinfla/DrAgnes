<script lang="ts">
	import MelaPanel from '$lib/components/MelaPanel.svelte';
	import DisclaimerGate from '$lib/components/DisclaimerGate.svelte';
</script>

<DisclaimerGate>
	<div class="flex h-dvh w-full flex-col bg-[#0a0a0f] overflow-hidden">
		<header class="flex shrink-0 items-center justify-between px-5 py-3 bg-[#0a0a0f]/95 backdrop-blur-md border-b border-white/[0.04]">
			<div class="flex items-center gap-2.5">
				<img src="/icons/icon-32x32.png" alt="Mela" class="h-8 w-8 rounded-full object-cover ring-1 ring-white/[0.08]" />
				<span class="text-[15px] font-semibold text-white tracking-tight">Mela</span>
				<span class="text-[13px] text-teal-400 font-semibold">v1.0.2</span>
			</div>
			<span class="text-[10px] text-gray-500 uppercase tracking-widest font-medium">Research Preview</span>
		</header>
		<main class="flex-1 overflow-hidden">
			<MelaPanel />
		</main>
	</div>
</DisclaimerGate>
