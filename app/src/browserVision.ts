/**
 * Browser-only vision stack helpers (GitHub Pages / static hosting).
 * No Node APIs — canvas, WebGL/WASM TF.js, getUserMedia, Web Workers only.
 */

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

export type BrowserVisionSupport = {
  ok: boolean;
  secureContext: boolean;
  mediaDevices: boolean;
  worker: boolean;
  canvas: boolean;
  webgl: boolean;
  message?: string;
};

let tfBackendPromise: Promise<string> | null = null;

/** Gate camera + COCO before requesting getUserMedia. */
export const checkBrowserVisionSupport = (): BrowserVisionSupport => {
  const secureContext =
    typeof window !== "undefined" &&
    (window.isSecureContext === true || location.hostname === "localhost");
  const mediaDevices = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
  const worker = typeof Worker !== "undefined";
  const canvas =
    typeof document !== "undefined" &&
    !!document.createElement("canvas").getContext("2d", { willReadFrequently: true });
  let webgl = false;
  try {
    const c = document.createElement("canvas");
    webgl = !!(c.getContext("webgl") || c.getContext("experimental-webgl"));
  } catch {
    webgl = false;
  }

  if (!secureContext) {
    return {
      ok: false,
      secureContext,
      mediaDevices,
      worker,
      canvas,
      webgl,
      message:
        "מצלמה ו-AI חזותי דורשים HTTPS (למשל GitHub Pages). localhost עובד לפיתוח.",
    };
  }
  if (!mediaDevices) {
    return {
      ok: false,
      secureContext,
      mediaDevices,
      worker,
      canvas,
      webgl,
      message: "הדפדפן לא תומך במצלמה (getUserMedia).",
    };
  }
  if (!worker) {
    return {
      ok: false,
      secureContext,
      mediaDevices,
      worker,
      canvas,
      webgl,
      message: "Web Workers לא נתמכים — Gemma לא יכול לרוץ.",
    };
  }
  if (!canvas) {
    return {
      ok: false,
      secureContext,
      mediaDevices,
      worker,
      canvas,
      webgl,
      message: "Canvas 2D לא זמין — ניתוח פריימים לא אפשרי.",
    };
  }

  return { ok: true, secureContext, mediaDevices, worker, canvas, webgl };
};

/** Init TF.js for COCO-SSD — CPU when camera+Gemma share the machine (avoids WebGL/WebGPU fights). */
export const ensureTfBackend = async (preferCpu = false): Promise<string> => {
  if (tfBackendPromise && !preferCpu) return tfBackendPromise;
  if (preferCpu) tfBackendPromise = null;

  tfBackendPromise = (async () => {
    await tf.ready();
    if (preferCpu) {
      await tf.setBackend("cpu");
      await tf.ready();
      return tf.getBackend();
    }
    const current = tf.getBackend();
    if (current) return current;
    try {
      await tf.setBackend("webgl");
      await tf.ready();
      return tf.getBackend();
    } catch (e) {
      console.warn("[GROVEE] WebGL TF backend failed, using CPU", e);
      await tf.setBackend("cpu");
      await tf.ready();
      return tf.getBackend();
    }
  })();
  return tfBackendPromise;
};

export const resetTfBackendCache = (): void => {
  tfBackendPromise = null;
};
