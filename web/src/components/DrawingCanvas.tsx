import { useRef, useEffect, useState, useCallback } from 'react';

interface DrawingCanvasProps {
  onPredict: (imageData: number[]) => void;
  onStroke?: () => void;
  disabled?: boolean;
  isLoading?: boolean;
}

const CANVAS_SIZE = 280;
const GRID_SIZE = 28;

export function DrawingCanvas({ 
  onPredict, 
  onStroke,
  disabled = false,
  isLoading = false 
}: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasContent, setHasContent] = useState(false);
  const lastPredictRef = useRef<number[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Initialize with black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  const getCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if ('touches' in e) {
      const touch = e.touches[0];
      return {
        x: (touch.clientX - rect.left) * scaleX,
        y: (touch.clientY - rect.top) * scaleY,
      };
    }

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    if (disabled) return;
    setIsDrawing(true);
    const { x, y } = getCoordinates(e);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) {
      ctx.beginPath();
      ctx.moveTo(x, y);
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || disabled) return;

    const { x, y } = getCoordinates(e);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) {
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 20;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
      setHasContent(true);
    }
  };

  const stopDrawing = () => {
    if (isDrawing && hasContent) {
      // Trigger prediction when stroke ends
      const imageData = getImageData();
      // Only predict if content changed meaningfully
      if (imageData.some((v, i) => Math.abs(v - (lastPredictRef.current[i] || 0)) > 0.01)) {
        lastPredictRef.current = imageData;
        onPredict(imageData);
        onStroke?.();
      }
    }
    setIsDrawing(false);
  };

  const getImageData = useCallback((): number[] => {
    const canvas = canvasRef.current;
    if (!canvas) return [];

    const ctx = canvas.getContext('2d');
    if (!ctx) return [];

    // Create a temporary canvas for downscaling
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = GRID_SIZE;
    tempCanvas.height = GRID_SIZE;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return [];

    // Draw scaled down version
    tempCtx.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);

    // Get pixel data
    const imageData = tempCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
    const pixels: number[] = [];

    // Convert to grayscale values (0-1)
    for (let i = 0; i < imageData.data.length; i += 4) {
      // Use red channel (grayscale image has same RGB values)
      const value = imageData.data[i] / 255;
      pixels.push(value);
    }

    return pixels;
  }, []);

  const handleClear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setHasContent(false);
    lastPredictRef.current = [];
  };

  // Get theme-aware border color
  const isDark = typeof document !== 'undefined' 
    ? document.documentElement.getAttribute('data-theme') !== 'light'
    : true;
  const borderColor = isDark ? '#444' : '#ccc';

  return (
    <div className="drawing-canvas-container">
      {isLoading && (
        <div className="canvas-loading-overlay">
          <div className="mini-spinner"></div>
          <span>Analyzing...</span>
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="drawing-canvas"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
        style={{
          border: `2px solid ${borderColor}`,
          borderRadius: '8px',
          cursor: disabled ? 'not-allowed' : 'crosshair',
          touchAction: 'none',
        }}
      />
      <div className="canvas-buttons">
        <button onClick={handleClear} disabled={disabled} className="clear-btn">
          🗑️ Clear
        </button>
      </div>
    </div>
  );
}
