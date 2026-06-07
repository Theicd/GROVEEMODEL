# GROVEE Model WebGPU

Browser UI similar to the Gemma-4 WebGPU demo, running ONNX models locally on the user's machine using WebGPU (or WASM fallback).

## Features

- Single local model: **Gemma 4 E2B** (`onnx-community/gemma-4-E2B-it-ONNX`).
- Load model weights directly from Hugging Face (no server inference API).
- Chat UI with token streaming from a Web Worker.
- **Vision:** attach images (file, paste, drag-drop) — Gemma 4 describes them locally.
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

- **Repo root `index.html`** only redirects to **`docs/index.html`** (production bundle). No `/src` paths — works even if Pages uses **`main` / `/ (root)`**.
- **Vite** uses **`app/index.html`** + **`app/src/`** for `npm run dev` and `npm run build`.
- **Build uses `base: './'`** so JS/CSS load as `./assets/...` from `docs/index.html`. A fixed `/GROVEEMODEL/` base was wrong: the browser requested `/GROVEEMODEL/assets/` while files sit under **`/GROVEEMODEL/docs/assets/`** → blank page.
- The folder **`docs/`** is updated by **`sync-docs-folder.yml`** after each push (copy of `dist/`).

Optional: set Pages to **`/docs`** so the site root is the bundle directly (no redirect). **GitHub Actions** deploy is also supported (`deploy-pages.yml`).

Console warnings like `Permissions-Policy ... browsing-topics` come from **github.io** response headers, not from this app; they can be ignored.

The repo has a **root `.nojekyll`** file so GitHub Pages does not run Jekyll (which can hide or mishandle static folders like `docs/`).

## Model

This build uses one ONNX model only:

- `onnx-community/gemma-4-E2B-it-ONNX` (q4, Transformers.js)

Constants live in `app/src/App.tsx` and `app/src/model.worker.ts`.
