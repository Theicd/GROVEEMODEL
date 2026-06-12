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
  let srcW = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  let srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!srcW || !srcH) {
    srcW = source instanceof HTMLVideoElement ? source.clientWidth : source.width;
    srcH = source instanceof HTMLVideoElement ? source.clientHeight : source.height;
  }
  if (!srcW || !srcH) {
    srcW = maxWidth;
    srcH = Math.round(maxWidth * 0.75);
  }
  const ratio = srcW / srcH;
  canvas.width = maxWidth;
  canvas.height = Math.round(maxWidth / ratio);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}
