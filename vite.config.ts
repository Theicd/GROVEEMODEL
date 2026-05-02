import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// https://vite.dev/config/
export default defineConfig({
  root: path.join(__dirname, "app"),
  publicDir: path.join(__dirname, "public"),
  // Relative so it works at /GROVEEMODEL/docs/ (files live under docs/assets/) and if Pages uses /docs as site root.
  base: "./",
  plugins: [react()],
  build: {
    outDir: path.join(__dirname, "dist"),
    emptyOutDir: true,
  },
});
