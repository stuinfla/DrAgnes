<script lang="ts">
	import { onMount } from "svelte";
	import type { Snippet } from "svelte";

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	const STORAGE_KEY = "mela-disclaimer-accepted";

	let accepted: boolean = $state(false);
	let mounted: boolean = $state(false);

	onMount(() => {
		accepted = localStorage.getItem(STORAGE_KEY) === "true";
		mounted = true;
	});

	function handleAccept() {
		localStorage.setItem(STORAGE_KEY, "true");
		accepted = true;
	}
</script>

{#if !mounted}
	<!-- Prevent flash: show nothing until we check localStorage -->
{:else if !accepted}
	<!-- Full-screen disclaimer overlay -->
	<div class="fixed inset-0 z-[9999] flex items-center justify-center bg-[#0a0a0f] p-4">
		<div class="w-full max-w-md rounded-3xl border border-white/[0.08] bg-[#12121a] px-6 py-8 shadow-2xl shadow-black/60">
			<!-- Header -->
			<div class="flex items-center gap-3 mb-6">
				<div class="flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/15">
					<svg class="h-5 w-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
						<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
					</svg>
				</div>
				<h2 class="text-lg font-semibold text-white tracking-tight">Important Notice</h2>
			</div>

			<!-- Disclaimer statements -->
			<div class="space-y-4 text-[13px] leading-relaxed text-gray-300">
				<p>
					Mela is an <span class="font-medium text-white">educational skin awareness tool</span>.
					It is <span class="font-medium text-amber-300">NOT a medical device</span> and is
					<span class="font-medium text-amber-300">NOT FDA-cleared</span>.
				</p>

				<p>
					Mela uses AI pattern analysis to help you learn about your skin. It does
					<span class="font-medium text-white">NOT</span> diagnose, screen for, or detect any disease.
					The pattern analysis results are for educational and informational purposes only.
				</p>

				<p>
					Mela does <span class="font-medium text-white">NOT</span> replace professional medical
					evaluation. If you have concerns about your skin, see a qualified healthcare provider.
				</p>

				<p class="text-gray-400 text-[12px] border-t border-white/[0.06] pt-4">
					By proceeding, you acknowledge that you understand these limitations and will not
					rely on Mela for any medical decision.
				</p>
			</div>

			<!-- Actions -->
			<div class="mt-6 flex flex-col gap-3">
				<button
					onclick={handleAccept}
					class="w-full rounded-full bg-teal-600 px-6 py-3.5 text-[15px] font-semibold text-white shadow-lg shadow-teal-600/20 hover:bg-teal-500 active:scale-[0.98] transition-all"
				>
					I Understand
				</button>
				<a
					href="/welcome"
					class="block w-full rounded-full border border-white/[0.08] bg-white/[0.03] px-6 py-3 text-center text-sm font-medium text-gray-400 hover:bg-white/[0.06] hover:text-gray-300 transition-all"
				>
					Learn More
				</a>
			</div>
		</div>
	</div>
{:else}
	{@render children()}
{/if}
