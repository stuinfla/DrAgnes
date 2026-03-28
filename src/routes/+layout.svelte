<script>
	import { onMount } from 'svelte';
	import '../app.css';
	let { children } = $props();

	onMount(() => {
		if ('serviceWorker' in navigator) {
			navigator.serviceWorker.register('/sw.js').catch(() => {});
			navigator.serviceWorker.register('/sw-model-cache.js', { scope: '/models/' }).catch(() => {});
		}
	});
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-950">
	{@render children()}
</div>
