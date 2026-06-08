import { useEffect, useRef } from "react";

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  s: number;
};

export function IntroCanvas({ contained = false }: { contained?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let raf = 0;
    const particles: Particle[] = [];

    const measure = () => {
      if (contained) {
        const parent = canvas.parentElement;
        if (!parent) return { width: 0, height: 0 };
        return { width: parent.clientWidth, height: parent.clientHeight };
      }
      return { width: window.innerWidth, height: window.innerHeight };
    };

    const seed = () => {
      particles.length = 0;
      const count = Math.max(18, Math.min(50, Math.floor((w * h) / 12000)));
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          s: Math.random() * 2,
        });
      }
    };

    const resize = () => {
      const { width, height } = measure();
      if (width <= 0 || height <= 0) return;
      const changed = width !== w || height !== h;
      w = canvas.width = width;
      h = canvas.height = height;
      if (changed) seed();
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;

        ctx.fillStyle = "rgba(0, 243, 255, 0.35)";
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.s, 0, Math.PI * 2);
        ctx.fill();

        for (let j = i + 1; j < particles.length; j++) {
          const q = particles[j];
          const dx = p.x - q.x;
          const dy = p.y - q.y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < 100) {
            ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 - d / 1000})`;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(draw);
    };

    resize();
    if (w > 0 && h > 0) seed();
    draw();

    let ro: ResizeObserver | undefined;
    if (contained) {
      const parent = canvas.parentElement;
      if (parent) {
        ro = new ResizeObserver(resize);
        ro.observe(parent);
      }
    } else {
      window.addEventListener("resize", resize);
    }

    return () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
      if (!contained) window.removeEventListener("resize", resize);
    };
  }, [contained]);

  return <canvas ref={canvasRef} className="bg-canvas" aria-hidden="true" />;
}
