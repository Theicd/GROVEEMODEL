# GROVEE Model WebGPU

Browser UI similar to the Gemma-4 WebGPU demo, running ONNX models locally on the user's machine using WebGPU (or WASM fallback).

## Features

- Load model files directly from Hugging Face (no server inference API).
- Chat UI with token streaming from a Web Worker.
- WebGPU first, automatic WASM fallback.
- Ready for static hosting (GitHub Pages / Vercel / Netlify / HF Spaces).

## Tech stack

- React + TypeScript + Vite
- `@huggingface/transformers` (Transformers.js)
- ONNX Runtime Web (brought by Transformers.js)

## Quick start

```bash
npm install
npm run dev
```

Open: `http://127.0.0.1:4173`

## Production build

```bash
npm run lint
npm run build
```

## Deploy notes

- You only host the frontend files (`dist/`), not the full model weights.
- Models are downloaded from Hugging Face to the browser cache on first use.
- For best browser performance, prefer ONNX models with quantized weights (`q4`/`q8`).
- Some advanced browser optimizations may require custom response headers (COOP/COEP), which are easier on Vercel/Netlify/HF Spaces than GitHub Pages.

### GitHub Pages (this repo)

The file **`index.html` in the repo root is for Vite dev** (`npm run dev`) and points at `/src/main.tsx`.  
If GitHub Pages is set to **“Deploy from branch `main` / folder `/ (root)`”**, the live site will 404 on `/src/main.tsx` and look blank.

**Do this once:** Repository **Settings → Pages → Build and deployment**

1. **Source:** *Deploy from a branch*
2. **Branch:** `main`
3. **Folder:** **`/docs`** (not `/ (root)`)

The production bundle is copied into **`docs/`** on each push (workflow `sync-docs-folder.yml`).  
Alternatively set **Source: GitHub Actions** and use `Deploy to GitHub Pages` workflow (artifact = `dist/` only).

Console warnings like `Permissions-Policy ... browsing-topics` come from **github.io** response headers, not from this app; they can be ignored.

The repo has a **root `.nojekyll`** file so GitHub Pages does not run Jekyll (which can hide or mishandle static folders like `docs/`).

## Model IDs you can try

- `onnx-community/gemma-4-E2B-it-ONNX`
- `onnx-community/Qwen3-0.6B-ONNX`

You can add more supported ONNX models by editing `MODEL_OPTIONS` in `src/App.tsx`.
