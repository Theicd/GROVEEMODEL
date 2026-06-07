import { useEffect, useRef } from "react";
import type { VisionResult } from "./vision-lab/core/types";
import { HAND_CONNECTIONS, POSE_CONNECTIONS } from "./vision-lab/core/types";

const COLORS = ["#22d3ee", "#34d399", "#a78bfa", "#f472b6", "#fbbf24"];

const flipX = (x: number, width: number): number => 1 - x - width;

export function VisionInspectorFeed({
  videoRef,
  result,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  result: VisionResult;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resultRef = useRef(result);
  resultRef.current = result;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let raf = 0;
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

      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      for (let i = 0; i < r.objects.length; i++) {
        const obj = r.objects[i];
        const color = COLORS[i % COLORS.length];
        const x = flipX(obj.bbox.x, obj.bbox.width) * canvas.width;
        const y = obj.bbox.y * canvas.height;
        const w = obj.bbox.width * canvas.width;
        const h = obj.bbox.height * canvas.height;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = color;
        ctx.font = "14px monospace";
        ctx.fillText(`${obj.displayLabel} ${Math.round(obj.confidence * 100)}%`, x, y - 4);
      }

      if (r.poseLandmarks.length) {
        ctx.strokeStyle = "#34d399";
        ctx.lineWidth = 2;
        for (const [a, b] of POSE_CONNECTIONS) {
          const p1 = r.poseLandmarks[a];
          const p2 = r.poseLandmarks[b];
          if (!p1 || !p2) continue;
          ctx.beginPath();
          ctx.moveTo(flipX(p1.x, 0) * canvas.width, p1.y * canvas.height);
          ctx.lineTo(flipX(p2.x, 0) * canvas.width, p2.y * canvas.height);
          ctx.stroke();
        }
      }

      for (const hand of r.hands) {
        ctx.strokeStyle = "#a78bfa";
        ctx.lineWidth = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
          const p1 = hand.landmarks[a];
          const p2 = hand.landmarks[b];
          ctx.beginPath();
          ctx.moveTo(flipX(p1.x, 0) * canvas.width, p1.y * canvas.height);
          ctx.lineTo(flipX(p2.x, 0) * canvas.width, p2.y * canvas.height);
          ctx.stroke();
        }
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [videoRef]);

  return <canvas ref={canvasRef} className="vision-inspector-canvas" aria-label="Live vision feed" />;
}
