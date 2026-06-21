import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { devRealityPlugin } from "./vite-plugins/devReality";
import { aisStreamProxyPlugin } from "./vite-plugins/aisStreamProxy";
import { tavilyProxyPlugin } from "./vite-plugins/tavilyProxy";
import { scavioProxyPlugin } from "./vite-plugins/scavioProxy";
import { fetchProxyPlugin } from "./vite/fetchProxyPlugin";
import { translateProxyPlugin } from "./vite/translateProxyPlugin";
import { liveMediaFavoritesSyncPlugin } from "./vite-plugins/liveMediaFavoritesSync";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoName = path.basename(__dirname);

// https://vite.dev/config/
export default defineConfig({
  root: path.join(__dirname, "app"),
  publicDir: path.join(__dirname, "public"),
  // Default ./ for local dev. Pages from repo /docs/: npm run build:pages-docs (VITE_BASE=/GROVEEMODEL/docs/)
  // Pages from Actions artifact at site root: VITE_BASE=/GROVEEMODEL/
  base: process.env.VITE_BASE ?? "./",
  resolve: {
    alias: {
      "@grovee-news": path.join(__dirname, "app/src/groveeNews/engine"),
    },
  },
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["@huggingface/transformers", "onnxruntime-web"],
  },
  plugins: [
    react(),
    devRealityPlugin(),
    aisStreamProxyPlugin(),
    tavilyProxyPlugin(),
    scavioProxyPlugin(),
    fetchProxyPlugin(),
    translateProxyPlugin(),
    liveMediaFavoritesSyncPlugin(),
    {
      name: "grovee-dev-banner",
      configureServer(server) {
        server.httpServer?.once("listening", () => {
          const addr = server.httpServer?.address();
          const port = typeof addr === "object" && addr ? addr.port : 5180;
          console.log(
            `\n  ▶ GROVEEMODEL — HAL space intro (${repoName})\n  → http://127.0.0.1:${port}/\n  ✖ NOT GROVEEMODEL-main (old «טען מודל מקומי» UI)\n`,
          );
        });
      },
    },
  ],
  build: {
    outDir: path.join(__dirname, "dist"),
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("@tensorflow-models/pose-detection/dist/movenet")) return "movenet";
          if (id.includes("@tensorflow-models/pose-detection")) return "pose-detection";
          if (id.includes("@tensorflow-models/coco-ssd")) return "coco-ssd";
          if (id.includes("onnxruntime-web")) return "onnxruntime";
          if (id.includes("@mediapipe/tasks-vision")) return "mediapipe-vision";
          if (id.includes("pdfjs-dist")) return "pdfjs";
        },
      },
    },
  },
});
