import { useEffect, useRef } from "react";
import { getFingerState } from "./vision-lab/analyzers/GestureRecognizer";
import type { VisionResult } from "./vision-lab/core/types";
import { HAND_CONNECTIONS, POSE_CONNECTIONS } from "./vision-lab/core/types";

const COLORS = ["#22d3ee", "#34d399", "#a78bfa", "#f472b6", "#fbbf24"];

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

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      for (let i = 0; i < r.objects.length; i++) {
        const obj = r.objects[i];
        const color = COLORS[i % COLORS.length];
        const x = obj.bbox.x * canvas.width;
        const y = obj.bbox.y * canvas.height;
        const w = obj.bbox.width * canvas.width;
        const h = obj.bbox.height * canvas.height;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = color;
        ctx.font = "12px monospace";
        ctx.fillText(`${obj.displayLabel} ${Math.round(obj.confidence * 100)}%`, x, Math.max(12, y - 2));
      }

      if (r.poseLandmarks.length) {
        ctx.strokeStyle = "#34d399";
        ctx.lineWidth = 2;
        for (const [a, b] of POSE_CONNECTIONS) {
          const p1 = r.poseLandmarks[a];
          const p2 = r.poseLandmarks[b];
          if (!p1 || !p2) continue;
          ctx.beginPath();
          ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
          ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
          ctx.stroke();
        }
        for (const lm of r.poseLandmarks) {
          ctx.fillStyle = "#6ee7b7";
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      for (const hand of r.hands) {
        const { count } = getFingerState(hand);
        ctx.strokeStyle = "#a78bfa";
        ctx.lineWidth = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
          const p1 = hand.landmarks[a];
          const p2 = hand.landmarks[b];
          ctx.beginPath();
          ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
          ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
          ctx.stroke();
        }
        for (const lm of hand.landmarks) {
          ctx.fillStyle = "#c4b5fd";
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, Math.PI * 2);
          ctx.fill();
        }
        const wrist = hand.landmarks[0];
        if (wrist) {
          ctx.fillStyle = "#e9d5ff";
          ctx.font = "bold 11px monospace";
          ctx.fillText(
            `${hand.handedness} · ${count}f`,
            wrist.x * canvas.width,
            Math.max(11, wrist.y * canvas.height - 6),
          );
        }
      }

      for (const face of r.faces) {
        const x = face.bbox.x * canvas.width;
        const y = face.bbox.y * canvas.height;
        const w = face.bbox.width * canvas.width;
        const h = face.bbox.height * canvas.height;
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "#fbbf24";
        ctx.font = "10px monospace";
        const emo = r.emotion?.dominant ? ` · ${r.emotion.dominant}` : "";
        ctx.fillText(`Face #${face.id}${emo}`, x, Math.max(10, y - 2));
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [videoRef]);

  return <canvas ref={canvasRef} className="vi-canvas" aria-label="Live vision feed" />;
}
