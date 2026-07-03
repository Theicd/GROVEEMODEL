import * as THREE from "three";

const W = 256;
const H = 96;

export type NeoLabelSprite = {
  sprite: THREE.Sprite;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
};

export function createNeoLabelSprite(): NeoLabelSprite {
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      sizeAttenuation: true,
    }),
  );
  sprite.scale.set(0.58, 0.22, 1);
  sprite.renderOrder = 10;
  return { sprite, canvas, ctx };
}

export function drawNeoLabel(
  label: NeoLabelSprite,
  lines: { title: string; dist: string; speed: string; eta: string },
  color: string,
) {
  const { ctx, canvas, sprite } = label;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(0,8,18,0.72)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  roundRect(ctx, 4, 4, canvas.width - 8, canvas.height - 8, 10);
  ctx.fill();
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.fillStyle = color;
  ctx.font = "bold 22px Segoe UI, Tahoma, sans-serif";
  ctx.fillText(lines.dist, canvas.width / 2, 34);
  ctx.font = "16px Segoe UI, Tahoma, sans-serif";
  ctx.fillStyle = "#e8f4ff";
  ctx.fillText(`${lines.speed}  ·  ${lines.eta}`, canvas.width / 2, 58);
  ctx.fillStyle = "#9ec8dd";
  ctx.font = "13px Segoe UI, Tahoma, sans-serif";
  ctx.fillText(lines.title, canvas.width / 2, 78);
  (sprite.material as THREE.SpriteMaterial).map!.needsUpdate = true;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
