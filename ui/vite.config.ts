import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Dev : lancer le coeur avec `python -m prophet_studio --dev --token dev --no-browser`, puis `npm run dev`
// (VITE_PROPHET_TOKEN=dev). En production l'interface est servie par le coeur ou embarquee par Tauri.
const core = process.env.PROPHET_CORE_URL ?? "http://127.0.0.1:7878";

export default defineConfig({
  plugins: [svelte()],
  base: "./",
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: core, ws: true, changeOrigin: false },
      "/v1": { target: core },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    cssCodeSplit: true,
    reportCompressedSize: false,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("highlight.js")) return "hljs";
          if (id.includes("marked") || id.includes("dompurify")) return "markdown";
        },
      },
    },
  },
});
