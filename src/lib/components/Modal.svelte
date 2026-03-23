<script lang="ts">
	import type { Snippet } from "svelte";

	interface Props {
		onclose: () => void;
		width?: string;
		children: Snippet;
	}

	let { onclose, width = "!max-w-[500px]", children }: Props = $props();

	function handleBackdropClick(e: MouseEvent) {
		if (e.target === e.currentTarget) {
			onclose();
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === "Escape") {
			onclose();
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
<div
	class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
	onclick={handleBackdropClick}
>
	<div class="relative mx-4 w-full max-w-lg rounded-2xl border border-gray-700 bg-gray-900 shadow-2xl {width}">
		<button
			onclick={onclose}
			class="absolute right-3 top-3 rounded-full p-1 text-gray-400 hover:bg-gray-800 hover:text-gray-200"
			aria-label="Close"
		>
			<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
			</svg>
		</button>
		{@render children()}
	</div>
</div>
