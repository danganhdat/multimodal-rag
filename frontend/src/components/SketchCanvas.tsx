import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  onChange: (base64: string | null) => void;
}

interface Point {
  x: number;
  y: number;
}

export default function SketchCanvas({ onChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [penSize, setPenSize] = useState(3);
  const strokesRef = useRef<Point[][]>([]);
  const currentStrokeRef = useRef<Point[]>([]);

  const getCanvasSize = useCallback(() => {
    const container = containerRef.current;
    if (!container) return 300;
    return Math.min(container.clientWidth, 300);
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#000000";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = penSize;

    for (const stroke of strokesRef.current) {
      if (stroke.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i].x, stroke[i].y);
      }
      ctx.stroke();
    }
  }, [penSize]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const size = getCanvasSize();
    canvas.width = size;
    canvas.height = size;
    redraw();
  }, [getCanvasSize, redraw]);

  const getPos = (e: React.MouseEvent | React.TouchEvent): Point => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if ("touches" in e) {
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

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDrawing(true);
    const pos = getPos(e);
    currentStrokeRef.current = [pos];
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pos = getPos(e);
    const prev = currentStrokeRef.current[currentStrokeRef.current.length - 1];
    currentStrokeRef.current.push(pos);

    ctx.strokeStyle = "#000000";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = penSize;
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const endDraw = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    if (currentStrokeRef.current.length > 0) {
      strokesRef.current.push([...currentStrokeRef.current]);
      currentStrokeRef.current = [];
      emitChange();
    }
  };

  const emitChange = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (strokesRef.current.length === 0) {
      onChange(null);
      return;
    }
    const dataUrl = canvas.toDataURL("image/png");
    const base64 = dataUrl.split(",")[1];
    onChange(base64);
  };

  const handleClear = () => {
    strokesRef.current = [];
    currentStrokeRef.current = [];
    redraw();
    onChange(null);
  };

  const handleUndo = () => {
    strokesRef.current.pop();
    redraw();
    emitChange();
  };

  return (
    <div ref={containerRef}>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          maxWidth: 300,
          aspectRatio: "1",
          border: "1px solid #dbdbdb",
          borderRadius: 4,
          cursor: "crosshair",
          touchAction: "none",
        }}
        onMouseDown={startDraw}
        onMouseMove={draw}
        onMouseUp={endDraw}
        onMouseLeave={endDraw}
        onTouchStart={startDraw}
        onTouchMove={draw}
        onTouchEnd={endDraw}
      />
      <div className="is-flex is-align-items-center mt-2" style={{ gap: "0.5rem" }}>
        <button className="button is-small" onClick={handleUndo} disabled={strokesRef.current.length === 0}>
          Undo
        </button>
        <button className="button is-small" onClick={handleClear}>
          Clear
        </button>
        <input
          type="range"
          min={1}
          max={10}
          value={penSize}
          onChange={(e) => setPenSize(Number(e.target.value))}
          style={{ flex: 1 }}
          title={`Pen size: ${penSize}`}
        />
        <span className="is-size-7 has-text-grey">{penSize}px</span>
      </div>
    </div>
  );
}
