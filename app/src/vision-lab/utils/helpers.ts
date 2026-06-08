export function getBaseUrl(): string {
  return import.meta.env.BASE_URL;
}

export function modelUrl(path: string): string {
  return `${getBaseUrl()}${path.replace(/^\//, '')}`;
}

export async function probeWebGpu(): Promise<boolean> {
  const gpu = (navigator as Navigator & {
    gpu?: { requestAdapter(): Promise<unknown> };
  }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

export function createOffscreenCanvas(
  source: HTMLVideoElement | HTMLCanvasElement,
  maxWidth = 640,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  const ratio = source instanceof HTMLVideoElement
    ? source.videoWidth / source.videoHeight
    : source.width / source.height;
  canvas.width = maxWidth;
  canvas.height = Math.round(maxWidth / ratio);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}
