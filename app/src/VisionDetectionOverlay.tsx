import { useEffect, useRef } from "react";
import { getFingerState } from "./vision-lab/analyzers/GestureRecognizer";
import type { VisionResult } from "./vision-lab/core/types";
import { HAND_CONNECTIONS, POSE_CONNECTIONS } from "./vision-lab/core/types";

const COLORS = ["#22d3ee", "#34d399", "#a78bfa", "#f472b6", "#fbbf24"];

/** Flip normalized x for selfie-mirrored video display. */
const flipX = (x: number, width: number, mirrored: boolean): number =>
  mirrored ? 1 - x - width : x;

export function VisionDetectionOverlay({
  videoRef,
  result,
  compact = false,
  mirrored = false,
  className = "",
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  result: VisionResult;
  compact?: boolean;
  mirrored?: boolean;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resultRef = useRef(result);
  resultRef.current = result;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let raf = 0;
    const lineWidth = compact ? 1.5 : 2;
    const fontSize = compact ? 9 : 14;
    const faceFontSize = compact ? 8 : 12;
    const dotRadius = compact ? 2.5 : 4;

    const draw = () => {
      const video = videoRef.current;
      const ctx = canvas.getContext("2d");
      if (!video || !ctx || video.readyState < 2) {
        raf = requestAnimationFrame(draw);
        return;
      }

      const r = resultRef.current;
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < r.objects.length; i++) {
        const obj = r.objects[i];
        const color = COLORS[i % COLORS.length];
        const nx = flipX(obj.bbox.x, obj.bbox.width, mirrored);
        const x = nx * canvas.width;
        const y = obj.bbox.y * canvas.height;
        const w = obj.bbox.width * canvas.width;
        const h = obj.bbox.height * canvas.height;

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = color;
        ctx.font = `${fontSize}px monospace`;
        ctx.fillText(
          `${obj.displayLabel} ${Math.round(obj.confidence * 100)}%`,
          x,
          Math.max(fontSize, y - 2),
        );
      }

      if (r.poseLandmarks.length) {
        ctx.strokeStyle = "#34d399";
        ctx.lineWidth = lineWidth;
        for (const [a, b] of POSE_CONNECTIONS) {
          const p1 = r.poseLandmarks[a];
          const p2 = r.poseLandmarks[b];
          if (!p1 || !p2) continue;
          ctx.beginPath();
          ctx.moveTo(flipX(p1.x, 0, mirrored) * canvas.width, p1.y * canvas.height);
          ctx.lineTo(flipX(p2.x, 0, mirrored) * canvas.width, p2.y * canvas.height);
          ctx.stroke();
        }
        for (const lm of r.poseLandmarks) {
          ctx.fillStyle = "#6ee7b7";
          ctx.beginPath();
          ctx.arc(
            flipX(lm.x, 0, mirrored) * canvas.width,
            lm.y * canvas.height,
            dotRadius,
            0,
            Math.PI * 2,
          );
          ctx.fill();
        }
      }

      for (const hand of r.hands) {
        const { count } = getFingerState(hand);
        ctx.strokeStyle = "#a78bfa";
        ctx.lineWidth = lineWidth;
        for (const [a, b] of HAND_CONNECTIONS) {
          const p1 = hand.landmarks[a];
          const p2 = hand.landmarks[b];
          ctx.beginPath();
          ctx.moveTo(flipX(p1.x, 0, mirrored) * canvas.width, p1.y * canvas.height);
          ctx.lineTo(flipX(p2.x, 0, mirrored) * canvas.width, p2.y * canvas.height);
          ctx.stroke();
        }
        for (const lm of hand.landmarks) {
          ctx.fillStyle = "#c4b5fd";
          ctx.beginPath();
          ctx.arc(
            flipX(lm.x, 0, mirrored) * canvas.width,
            lm.y * canvas.height,
            dotRadius,
            0,
            Math.PI * 2,
          );
          ctx.fill();
        }
        const wrist = hand.landmarks[0];
        if (wrist) {
          const labelX = flipX(wrist.x, 0, mirrored) * canvas.width;
          const labelY = wrist.y * canvas.height;
          ctx.fillStyle = "#e9d5ff";
          ctx.font = `bold ${fontSize}px monospace`;
          ctx.fillText(
            `${hand.handedness} · ${count}`,
            labelX,
            Math.max(fontSize, labelY - 6),
          );
        }
      }

      for (const face of r.faces) {
        const nx = flipX(face.bbox.x, face.bbox.width, mirrored);
        const x = nx * canvas.width;
        const y = face.bbox.y * canvas.height;
        const w = face.bbox.width * canvas.width;
        const h = face.bbox.height * canvas.height;
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "#fbbf24";
        ctx.font = `${faceFontSize}px monospace`;
        ctx.fillText(`Face #${face.id}`, x, Math.max(faceFontSize, y - 2));
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, compact, mirrored]);

  return (
    <canvas
      ref={canvasRef}
      className={`vision-detection-overlay ${className}`.trim()}
      aria-hidden="true"
    />
  );
}
