import adapterNode from '@sveltejs/adapter-node';
import adapterVercel from '@sveltejs/adapter-vercel';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const useVercel = process.env.VERCEL || process.env.ADAPTER === 'vercel';

/** @type {import('@sveltejs/kit').Config} */
export default {
	preprocess: vitePreprocess(),
	kit: {
		adapter: useVercel
			? adapterVercel({ runtime: 'nodejs22.x' })
			: adapterNode({ out: 'build' }),
		alias: {
			'$lib': 'src/lib'
		}
	}
};
