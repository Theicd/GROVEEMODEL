import { useEffect, useRef } from "react";

type Props = {
  active: boolean;
  /** Fade out after lock-on */
  fading?: boolean;
};

/** Analog TV snow while the cable box "tunes" a channel. */
export function TvStaticOverlay({ active, fading }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!active) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement!);

    const draw = () => {
      const { width, height } = canvas;
      if (width < 1 || height < 1) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const imageData = ctx.createImageData(width, height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const v = (Math.random() * 255) | 0;
        data[i] = v;
        data[i + 1] = v;
        data[i + 2] = v;
        data[i + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);

      ctx.fillStyle = "rgba(0,0,0,0.08)";
      for (let y = 0; y < height; y += 3) {
        ctx.fillRect(0, y, width, 1);
      }
      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [active]);

  if (!active && !fading) return null;

  return (
    <canvas
      ref={canvasRef}
      className={`lm-static-overlay${fading ? " lm-static-overlay--fade" : ""}`}
      aria-hidden="true"
    />
  );
}
