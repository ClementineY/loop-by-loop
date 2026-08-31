// @ts-check
import { defineConfig } from 'astro/config';
import svelte from '@astrojs/svelte';
import mdx from '@astrojs/mdx';

const runtime = /** @type {typeof globalThis & { process?: { env: Record<string, string | undefined> } }} */ (globalThis);
const productionSite = runtime.process?.env.SITE_URL;

// https://astro.build/config
export default defineConfig({
  ...(productionSite ? { site: productionSite } : {}),
  base: runtime.process?.env.BASE_PATH ?? '/',
  integrations: [svelte(), mdx()],
  markdown: {
    shikiConfig: { theme: 'github-dark-default' },
  },
});
