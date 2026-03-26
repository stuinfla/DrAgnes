/**
 * Type declarations for unplugin-icons virtual modules.
 *
 * unplugin-icons generates Svelte components at build time from icon sets
 * (e.g. ~icons/carbon/camera). These declarations tell TypeScript that
 * every ~icons/* import resolves to a valid Svelte component.
 */
declare module '~icons/*' {
	import { SvelteComponent } from 'svelte';
	export default class extends SvelteComponent<Record<string, never>> {}
}
