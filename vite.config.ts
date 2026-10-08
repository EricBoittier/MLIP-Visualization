import { defineConfig } from 'vite';
export default defineConfig({
  base: './',
  worker: { format: 'es' },
  server: { port: 5173 },
  build: { rollupOptions: { input: { main: 'index.html', md: 'md.html' } } },
});
