/** Camera stream lifecycle and frame capture for vision analysis. */

import { checkBrowserVisionSupport } from "./browserVision";

export type CameraStreamHandle = {
  stream: MediaStream;
  stop: () => void;
};

export const requestCameraStream = async (): Promise<CameraStreamHandle> => {
  const support = checkBrowserVisionSupport();
  if (!support.ok) {
    throw new Error(support.message ?? "Vision stack not supported in this browser");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("הדפדפן לא תומך במצלמה (נדרש HTTPS או localhost)");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "user" },
      width: { ideal: 640, max: 1280 },
      height: { ideal: 480, max: 720 },
      frameRate: { ideal: 15, max: 24 },
    },
    audio: false,
  });
  return {
    stream,
    stop: () => {
      for (const track of stream.getTracks()) track.stop();
    },
  };
};

export const attachStreamToVideo = async (
  video: HTMLVideoElement,
  stream: MediaStream,
): Promise<void> => {
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  video.setAttribute("playsinline", "true");
  await video.play();
};

export const captureVideoFrame = async (
  video: HTMLVideoElement,
  maxDim = 512,
  quality = 0.82,
): Promise<ArrayBuffer> => {
  if (video.readyState < 2 || video.videoWidth <= 0) {
    throw new Error("המצלמה עדיין לא מוכנה");
  }
  const scale = Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight));
  const w = Math.max(1, Math.round(video.videoWidth * scale));
  const h = Math.max(1, Math.round(video.videoHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas לא נתמך");
  ctx.drawImage(video, 0, 0, w, h);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("צילום נכשל"))), "image/jpeg", quality);
  });
  return blob.arrayBuffer();
};
