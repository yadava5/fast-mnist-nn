import { useEffect, useRef, useCallback, useMemo } from 'react';

interface NeuralNetVizProps {
  confidence: number[];
  prediction: number | null;
  isAnimating: boolean;
}

// Simplified visualization: show fewer nodes but represent the full network
const INPUT_SAMPLE = 16;    // Show 16 nodes representing 784
const HIDDEN_SAMPLE = 12;   // Show 12 nodes representing 100
const OUTPUT_NODES = 10;    // Show all 10 output nodes
const WIDTH = 400;
const HEIGHT = 300;
const LAYER_X = [60, 200, 340] as const; // X positions for input, hidden, output layers

interface Particle {
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  progress: number;
  layer: number; // 0 = input->hidden, 1 = hidden->output
  opacity: number;
}

type AnimationPhase = 'idle' | 'layer1' | 'layer2' | 'complete';

export function NeuralNetViz({ confidence, prediction, isAnimating }: NeuralNetVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);
  const particlesRef = useRef<Particle[]>([]);
  const phaseRef = useRef<AnimationPhase>('idle');
  const lastAnimatingRef = useRef(false);
  
  // Calculate node positions (memoized since they don't change)
  const nodePositions = useMemo(() => {
    const inputNodes = Array.from({ length: INPUT_SAMPLE }, (_, i) => ({
      x: LAYER_X[0],
      y: 30 + (i * (HEIGHT - 60)) / (INPUT_SAMPLE - 1),
    }));
    
    const hiddenNodes = Array.from({ length: HIDDEN_SAMPLE }, (_, i) => ({
      x: LAYER_X[1],
      y: 40 + (i * (HEIGHT - 80)) / (HIDDEN_SAMPLE - 1),
    }));
    
    const outputNodes = Array.from({ length: OUTPUT_NODES }, (_, i) => ({
      x: LAYER_X[2],
      y: 30 + (i * (HEIGHT - 60)) / (OUTPUT_NODES - 1),
    }));
    
    return { inputNodes, hiddenNodes, outputNodes };
  }, []);

  // Create particles for animation
  const createLayer1Particles = useCallback(() => {
    const { inputNodes, hiddenNodes } = nodePositions;
    const newParticles: Particle[] = [];
    
    for (let i = 0; i < 20; i++) {
      const sourceNode = inputNodes[Math.floor(Math.random() * inputNodes.length)];
      const targetNode = hiddenNodes[Math.floor(Math.random() * hiddenNodes.length)];
      newParticles.push({
        x: sourceNode.x,
        y: sourceNode.y,
        targetX: targetNode.x,
        targetY: targetNode.y,
        progress: Math.random() * 0.3,
        layer: 0,
        opacity: 1,
      });
    }
    return newParticles;
  }, [nodePositions]);

  const createLayer2Particles = useCallback((pred: number | null) => {
    const { hiddenNodes, outputNodes } = nodePositions;
    const layer2Particles: Particle[] = [];
    
    for (let i = 0; i < 15; i++) {
      const sourceNode = hiddenNodes[Math.floor(Math.random() * hiddenNodes.length)];
      const targetIdx = pred !== null && Math.random() > 0.4 
        ? pred 
        : Math.floor(Math.random() * OUTPUT_NODES);
      const targetNode = outputNodes[targetIdx];
      
      layer2Particles.push({
        x: sourceNode.x,
        y: sourceNode.y,
        targetX: targetNode.x,
        targetY: targetNode.y,
        progress: Math.random() * 0.3,
        layer: 1,
        opacity: 1,
      });
    }
    return layer2Particles;
  }, [nodePositions]);

  // Animation loop - uses refs to avoid setState in effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let layer2Timeout: ReturnType<typeof setTimeout> | null = null;
    let completeTimeout: ReturnType<typeof setTimeout> | null = null;

    // Handle animation start
    if (isAnimating && !lastAnimatingRef.current) {
      phaseRef.current = 'layer1';
      particlesRef.current = createLayer1Particles();
      
      layer2Timeout = setTimeout(() => {
        phaseRef.current = 'layer2';
        const remaining = particlesRef.current.filter(p => p.progress < 1);
        particlesRef.current = [...remaining, ...createLayer2Particles(prediction)];
      }, 400);
      
      completeTimeout = setTimeout(() => {
        phaseRef.current = 'complete';
        particlesRef.current = [];
      }, 1000);
    } else if (!isAnimating && lastAnimatingRef.current) {
      phaseRef.current = 'idle';
      particlesRef.current = [];
    }
    
    lastAnimatingRef.current = isAnimating;

    const render = () => {
      // Clear canvas
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
      
      const { inputNodes, hiddenNodes, outputNodes } = nodePositions;
      const currentPhase = phaseRef.current;
      const particles = particlesRef.current;
      
      // Get theme colors
      const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
      const lineColor = isDark ? 'rgba(102, 126, 234, 0.1)' : 'rgba(102, 126, 234, 0.15)';
      const nodeColor = isDark ? '#444' : '#ccc';
      const activeNodeColor = '#667eea';
      const textColor = isDark ? '#fff' : '#333';
      
      // Draw connections (faded)
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 0.5;
      
      // Input to hidden connections (sparse)
      inputNodes.forEach((input, i) => {
        hiddenNodes.forEach((hidden, j) => {
          if ((i + j) % 3 === 0) {
            ctx.beginPath();
            ctx.moveTo(input.x, input.y);
            ctx.lineTo(hidden.x, hidden.y);
            ctx.stroke();
          }
        });
      });
      
      // Hidden to output connections
      hiddenNodes.forEach((hidden, i) => {
        outputNodes.forEach((output, j) => {
          if ((i + j) % 2 === 0) {
            ctx.beginPath();
            ctx.moveTo(hidden.x, hidden.y);
            ctx.lineTo(output.x, output.y);
            ctx.stroke();
          }
        });
      });
      
      // Draw and update particles
      particles.forEach(particle => {
        if (particle.progress >= 0 && particle.progress <= 1) {
          const x = particle.x + (particle.targetX - particle.x) * particle.progress;
          const y = particle.y + (particle.targetY - particle.y) * particle.progress;
          
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(102, 126, 234, ${particle.opacity * (1 - particle.progress * 0.5)})`;
          ctx.fill();
          
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(102, 126, 234, ${particle.opacity * 0.3 * (1 - particle.progress * 0.5)})`;
          ctx.fill();
        }
        particle.progress += 0.03;
      });
      
      // Remove completed particles
      particlesRef.current = particles.filter(p => p.progress <= 1.2);
      
      // Draw input layer nodes
      inputNodes.forEach((node) => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = currentPhase !== 'idle' ? activeNodeColor : nodeColor;
        ctx.fill();
      });
      
      // Draw hidden layer nodes
      hiddenNodes.forEach((node) => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = currentPhase === 'layer2' || currentPhase === 'complete' 
          ? activeNodeColor 
          : nodeColor;
        ctx.fill();
      });
      
      // Draw output layer nodes with confidence
      outputNodes.forEach((node, i) => {
        const conf = confidence[i] || 0;
        const isWinner = prediction === i;
        const radius = isWinner ? 10 : 6 + conf * 4;
        
        if (isWinner && currentPhase === 'complete') {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 8, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
          ctx.fill();
        }
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        
        if (isWinner && (currentPhase === 'complete' || currentPhase === 'idle')) {
          ctx.fillStyle = '#667eea';
        } else if (conf > 0.1) {
          ctx.fillStyle = `rgba(102, 126, 234, ${0.3 + conf * 0.7})`;
        } else {
          ctx.fillStyle = nodeColor;
        }
        ctx.fill();
        
        ctx.fillStyle = textColor;
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i), node.x, node.y);
      });
      
      // Draw layer labels
      ctx.fillStyle = isDark ? '#666' : '#999';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      
      ctx.fillText('Input', LAYER_X[0], HEIGHT - 10);
      ctx.fillText('784', LAYER_X[0], HEIGHT - 22);
      
      ctx.fillText('Hidden', LAYER_X[1], HEIGHT - 10);
      ctx.fillText('100', LAYER_X[1], HEIGHT - 22);
      
      ctx.fillText('Output', LAYER_X[2], HEIGHT - 10);
      ctx.fillText('10', LAYER_X[2], HEIGHT - 22);
      
      animationRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (layer2Timeout) {
        clearTimeout(layer2Timeout);
      }
      if (completeTimeout) {
        clearTimeout(completeTimeout);
      }
    };
  }, [confidence, prediction, isAnimating, nodePositions, createLayer1Particles, createLayer2Particles]);

  return (
    <div className="neural-net-viz">
      <h3>🔗 Network Architecture</h3>
      <canvas 
        ref={canvasRef} 
        width={WIDTH} 
        height={HEIGHT}
        className="neural-net-canvas"
      />
      {prediction !== null && confidence.length > 0 && (
        <div className="viz-legend">
          <span className="legend-item">
            <span className="legend-dot active"></span>
            Predicted: {prediction} ({(confidence[prediction] * 100).toFixed(1)}%)
          </span>
        </div>
      )}
    </div>
  );
}
