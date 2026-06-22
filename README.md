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

## GROVEE-NEWS integration

Work-in-progress plan to merge the **GROVEE-NEWS** engine into this repo (replace legacy `news-rss`):

- [docs/GROVEE-NEWS-INTEGRATION-PLAN.md](docs/GROVEE-NEWS-INTEGRATION-PLAN.md) — full phases, QA, E2E flows
- [docs/GROVEE-NEWS-INTEGRATION-CHECKLIST.md](docs/GROVEE-NEWS-INTEGRATION-CHECKLIST.md) — short tracking checklist

## Quick start

**Use this folder only:** `GROVEEMODEL` (not `GROVEEMODEL-main` — that clone is deprecated and shows the old intro).

```bash
npm install
npm run dev
```

Open: **`http://127.0.0.1:5180/`** — top bar shows **`HAL·5180`**. Intro button: **«טען מודל לדפדפן»** (space / Earth fly-by).

### GROVEE Desktop (Windows end users)

For users without Node.js — full local install (UI + search):

```bash
npm run build:desktop
```

Produces `public/plugins/GroveDesktop-Setup-1.0.0.exe` (requires [Inno Setup 6](https://jrsoftware.org/isinfo.php) on the build machine) and `grove-desktop-win.zip`. The plugins panel download button points to the installer.

After install: desktop icon **GROVEE** → opens `http://127.0.0.1:5180` with search on `7000`.

If you see the old centered logo and **«טען מודל מקומי»**, you are on the wrong port, wrong folder, or stale GitHub Pages cache — hard-refresh or use the URL above.

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

- **Live site:** `https://theicd.github.io/GROVEEMODEL/docs/`
- **Repo root `index.html`** only redirects to **`docs/index.html`** (production bundle).
- **Update Pages after UI changes:** `npm run build:pages-docs` then commit and push **`docs/`**.
- **Vite** uses **`app/index.html`** + **`app/src/`** for `npm run dev` and `npm run build`.
- **Build uses `base: './'`** so JS/CSS load as `./assets/...` from `docs/index.html`.
- **`build:pages-docs`** wipes and recopies `dist/` → `docs/` and prunes stale hashed assets (one bundle only).

Optional: set Pages to **`/docs`** so the site root is the bundle directly (no redirect). **GitHub Actions** deploy is also supported (`deploy-pages.yml`).

Console warnings like `Permissions-Policy ... browsing-topics` come from **github.io** response headers, not from this app; they can be ignored.

The repo has a **root `.nojekyll`** file so GitHub Pages does not run Jekyll (which can hide or mishandle static folders like `docs/`).

## Model

This build uses one ONNX model only:

- `onnx-community/gemma-4-E2B-it-ONNX` (q4, Transformers.js)

Constants live in `app/src/App.tsx` and `app/src/model.worker.ts`.
