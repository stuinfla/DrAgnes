import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import Icons from 'unplugin-icons/vite';
import { readFileSync } from 'fs';

const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'));

export default defineConfig({
	plugins: [
		sveltekit(),
		Icons({ compiler: 'svelte' }),
	],
	define: {
		'__APP_VERSION__': JSON.stringify(pkg.version),
	},
	build: {
		rollupOptions: {
			external: ['@ruvector/cnn', 'onnxruntime-node', 'sharp']
		}
	},
	ssr: {
		external: ['@ruvector/cnn', 'onnxruntime-node', 'sharp']
	},
	test: {
		include: ['tests/**/*.test.ts', 'src/**/*.test.ts']
	}
});
