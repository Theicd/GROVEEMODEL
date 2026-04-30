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

## Model IDs you can try

- `onnx-community/gemma-4-E2B-it-ONNX`
- `onnx-community/Qwen3-0.6B-ONNX`

You can add more supported ONNX models by editing `MODEL_OPTIONS` in `src/App.tsx`.
