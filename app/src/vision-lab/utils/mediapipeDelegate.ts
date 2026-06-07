/** Prefer GPU like browser-vision-lab; fall back to CPU when WebGPU/GPU delegate is unavailable. */
export async function probeMediapipeGpu(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

export async function pickMediapipeDelegate(): Promise<"GPU" | "CPU"> {
  return (await probeMediapipeGpu()) ? "GPU" : "CPU";
}
