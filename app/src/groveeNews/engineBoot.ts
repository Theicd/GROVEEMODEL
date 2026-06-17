import { startEngine } from "./engine/engine/pipeline";

let bootPromise: Promise<void> | null = null;
let bootDone = false;

export function isGroveeNewsReady(): boolean {
  return bootDone;
}

export async function startGroveeNewsBoot(): Promise<void> {
  if (bootPromise) return bootPromise;
  bootPromise = (async () => {
    try {
      await startEngine();
      bootDone = true;
    } catch (err) {
      console.warn("[GROVEE-NEWS] engine boot failed", err);
      bootPromise = null;
    }
  })();
  return bootPromise;
}
