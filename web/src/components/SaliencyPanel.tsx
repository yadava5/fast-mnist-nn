import { useEffect, useRef } from 'react';

interface SaliencyPanelProps {
  /** Per-pixel gradient, length 784 (28 * 28). Arbitrary sign + magnitude. */
  inputGrad?: number[];
  /** Rendered pixel size (square canvas). Device-pixel-ratio aware. */
  size?: number;
}

const GRID = 28;

/**
 * Panel 1: 28x28 saliency overlay.
 *
 * - Cell color comes from a diverging colormap:
 *     input_grad[i] < 0 -> cool blue  oklch(0.7 0.15 220)
 *     input_grad[i] = 0 -> neutral    oklch(0.6 0 0 / 0.3)
 *     input_grad[i] > 0 -> warm red   oklch(0.65 0.22 25)
 * - Magnitude / max-magnitude maps to opacity so zero-grad pixels
 *   fade out and the network's most-attended pixels pop.
 * - Missing gradient -> uniform 20% opacity fallback.
 */
export function SaliencyPanel({ inputGrad, size = 220 }: SaliencyPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const present = !!inputGrad && inputGrad.length >= GRID * GRID;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const cell = size / GRID;

    if (!present) {
      ctx.fillStyle = 'oklch(0.6 0 0 / 0.2)';
      ctx.fillRect(0, 0, size, size);
      // Subtle grid cue so the panel reads as 28x28 even in fallback.
      ctx.strokeStyle = 'oklch(0.5 0 0 / 0.15)';
      ctx.lineWidth = 1;
      for (let i = 1; i < GRID; i++) {
        const p = i * cell;
        ctx.beginPath();
        ctx.moveTo(p, 0);
        ctx.lineTo(p, size);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, p);
        ctx.lineTo(size, p);
        ctx.stroke();
      }
      return;
    }

    const grad = inputGrad!;
    let maxMag = 0;
    for (let i = 0; i < GRID * GRID; i++) {
      const m = Math.abs(grad[i]);
      if (m > maxMag) maxMag = m;
    }
    if (maxMag === 0) maxMag = 1; // avoid divide-by-zero on flat grads

    for (let row = 0; row < GRID; row++) {
      for (let col = 0; col < GRID; col++) {
        const v = grad[row * GRID + col];
        const mag = Math.abs(v) / maxMag;
        if (mag === 0) continue;
        const color = v > 0 ? `oklch(0.65 0.22 25 / ${mag})` : `oklch(0.7 0.15 220 / ${mag})`;
        ctx.fillStyle = color;
        ctx.fillRect(col * cell, row * cell, cell, cell);
      }
    }
  }, [inputGrad, present, size]);

  return (
    <div className="activation-panel">
      <span className="activation-panel-label">Saliency</span>
      <canvas
        ref={canvasRef}
        className="activation-canvas"
        style={{ width: size, height: size }}
        aria-label="Input saliency overlay (28x28)"
      />
    </div>
  );
}
