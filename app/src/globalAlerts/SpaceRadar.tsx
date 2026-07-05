import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import * as THREE from "three";
import { estimateLiveDistLd } from "./neoLiveMetrics";
import {
  inferSpectralType,
  neoVisualSize,
  SPECTRAL_TYPES,
  type SpectralKey,
} from "./spaceObjectVisuals";
import type { GlobeAlertEvent } from "./types";

type Props = {
  events: GlobeAlertEvent[];
  focusedId?: string | null;
  visible?: boolean;
};

function radarAngleFromLon(lon: number): number {
  return ((lon + 180) * Math.PI) / 180;
}

export function SpaceRadar({ events, focusedId, visible = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const angleRef = useRef(0);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!visible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const neos = events.filter((e) => e.type === "neo");

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const r = w / 2 - 10;
      const rx = ctx;

      rx.fillStyle = "rgba(0,10,2,0.92)";
      rx.fillRect(0, 0, w, h);

      rx.strokeStyle = "rgba(0,255,80,0.12)";
      rx.lineWidth = 0.5;
      for (let i = 1; i <= 4; i++) {
        rx.beginPath();
        rx.arc(cx, cy, (r * i) / 4, 0, Math.PI * 2);
        rx.stroke();
      }
      rx.beginPath();
      rx.moveTo(cx, 10);
      rx.lineTo(cx, h - 10);
      rx.moveTo(10, cy);
      rx.lineTo(w - 10, cy);
      rx.stroke();

      angleRef.current += 0.025;
      const radarAngle = angleRef.current;
      rx.save();
      rx.beginPath();
      rx.moveTo(cx, cy);
      rx.arc(cx, cy, r, radarAngle - 0.5, radarAngle, false);
      rx.closePath();
      const sg = rx.createRadialGradient(cx, cy, 0, cx, cy, r);
      sg.addColorStop(0, "rgba(0,255,80,0.12)");
      sg.addColorStop(1, "rgba(0,255,80,0.01)");
      rx.fillStyle = sg;
      rx.fill();
      rx.restore();

      rx.strokeStyle = "rgba(0,255,80,0.6)";
      rx.lineWidth = 1.5;
      rx.beginPath();
      rx.moveTo(cx, cy);
      rx.lineTo(cx + Math.cos(radarAngle) * r, cy + Math.sin(radarAngle) * r);
      rx.stroke();

      rx.fillStyle = "#0066FF";
      rx.beginPath();
      rx.arc(cx, cy, 5, 0, Math.PI * 2);
      rx.fill();

      for (const ev of neos) {
        const liveLd = estimateLiveDistLd(ev);
        const rd = Math.min(liveLd / 6, 1) * (r - 12);
        const angle = radarAngleFromLon(ev.lon);
        const ax = cx + Math.cos(angle) * rd;
        const ay = cy + Math.sin(angle) * rd;
        const specKey = inferSpectralType(ev) as SpectralKey;
        const col = new THREE.Color(SPECTRAL_TYPES[specKey].color);
        const colHex = "#" + col.getHexString();
        const sz = Math.max(2, Math.min(8, neoVisualSize(ev.diameterKm) * 35));
        rx.fillStyle = colHex;
        rx.beginPath();
        rx.arc(ax, ay, sz, 0, Math.PI * 2);
        rx.fill();
        if (ev.isPha && !ev.showcaseNeo) {
          rx.strokeStyle = "#FF3333";
          rx.lineWidth = 1;
          rx.beginPath();
          rx.arc(ax, ay, sz + 3, 0, Math.PI * 2);
          rx.stroke();
        }
        if (ev.id === focusedId) {
          rx.strokeStyle = "#FFFFFF";
          rx.lineWidth = 1.5;
          rx.beginPath();
          rx.arc(ax, ay, sz + 5, 0, Math.PI * 2);
          rx.stroke();
        }
      }

      rx.fillStyle = "rgba(0,255,80,0.35)";
      rx.font = "7px monospace";
      rx.textAlign = "center";
      rx.fillText(`${neos.length} עצמים`, cx, h - 6);

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [events, focusedId, visible]);

  if (!visible) return null;

  return createPortal(
    <div className="space-radar-anchor" dir="ltr">
      <canvas ref={canvasRef} className="space-radar" width={200} height={200} aria-label="ראדר" />
    </div>,
    document.body,
  );
}
