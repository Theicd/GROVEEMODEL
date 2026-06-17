// @ts-nocheck
const STORAGE_KEY = "gn-ai-deep-read";



const listeners = new Set<() => void>();



function notify(): void {

  listeners.forEach((cb) => cb());

}



export function isAiDeepReadEnabled(): boolean {

  try {

    return localStorage.getItem(STORAGE_KEY) === "1";

  } catch {

    return false;

  }

}



export function setAiDeepReadEnabled(enabled: boolean): void {

  try {

    localStorage.setItem(STORAGE_KEY, enabled ? "1" : "0");

  } catch {

    /* ignore */

  }

  notify();

}



export function subscribeAiDeepRead(cb: () => void): () => void {

  listeners.add(cb);

  return () => listeners.delete(cb);

}


