import { useEffect, useRef, useState, useCallback } from 'react';

interface NeuralNetVizProps {
  confidence: number[];
  prediction: number | null;
  isAnimating: boolean;
}

// Simplified visualization: show fewer nodes but represent the full network
const INPUT_SAMPLE = 16;    // Show 16 nodes representing 784
const HIDDEN_SAMPLE = 12;   // Show 12 nodes representing 100
const OUTPUT_NODES = 10;    // Show all 10 output nodes

interface Particle {
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  progress: number;
  layer: number; // 0 = input->hidden, 1 = hidden->output
  opacity: number;
}

export function NeuralNetViz({ confidence, prediction, isAnimating }: NeuralNetVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [animationPhase, setAnimationPhase] = useState<'idle' | 'layer1' | 'layer2' | 'complete'>('idle');
  const animationRef = useRef<number | undefined>(undefined);

  const width = 400;
  const height = 300;
  const layerX = [60, 200, 340]; // X positions for input, hidden, output layers
  
  // Calculate node positions
  const getNodePositions = useCallback(() => {
    const inputNodes = Array.from({ length: INPUT_SAMPLE }, (_, i) => ({
      x: layerX[0],
      y: 30 + (i * (height - 60)) / (INPUT_SAMPLE - 1),
    }));
    
    const hiddenNodes = Array.from({ length: HIDDEN_SAMPLE }, (_, i) => ({
      x: layerX[1],
      y: 40 + (i * (height - 80)) / (HIDDEN_SAMPLE - 1),
    }));
    
    const outputNodes = Array.from({ length: OUTPUT_NODES }, (_, i) => ({
      x: layerX[2],
      y: 30 + (i * (height - 60)) / (OUTPUT_NODES - 1),
    }));
    
    return { inputNodes, hiddenNodes, outputNodes };
  }, [height]);

  // Trigger animation when isAnimating changes
  useEffect(() => {
    if (isAnimating) {
      setAnimationPhase('layer1');
      
      // Create particles for layer 1 (input -> hidden)
      const { inputNodes, hiddenNodes } = getNodePositions();
      const newParticles: Particle[] = [];
      
      // Create particles from random input nodes to random hidden nodes
      for (let i = 0; i < 20; i++) {
        const sourceNode = inputNodes[Math.floor(Math.random() * inputNodes.length)];
        const targetNode = hiddenNodes[Math.floor(Math.random() * hiddenNodes.length)];
        newParticles.push({
          x: sourceNode.x,
          y: sourceNode.y,
          targetX: targetNode.x,
          targetY: targetNode.y,
          progress: Math.random() * 0.3, // Stagger start
          layer: 0,
          opacity: 1,
        });
      }
      
      setParticles(newParticles);
      
      // Start layer 2 after delay
      setTimeout(() => {
        setAnimationPhase('layer2');
        const { hiddenNodes, outputNodes } = getNodePositions();
        const layer2Particles: Particle[] = [];
        
        // Create particles focusing on the predicted output
        for (let i = 0; i < 15; i++) {
          const sourceNode = hiddenNodes[Math.floor(Math.random() * hiddenNodes.length)];
          // Bias particles toward predicted output
          const targetIdx = prediction !== null && Math.random() > 0.4 
            ? prediction 
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
        
        setParticles(prev => [...prev.filter(p => p.progress < 1), ...layer2Particles]);
      }, 400);
      
      // Complete animation
      setTimeout(() => {
        setAnimationPhase('complete');
        setParticles([]);
      }, 1000);
    } else {
      setAnimationPhase('idle');
    }
  }, [isAnimating, prediction, getNodePositions]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      const { inputNodes, hiddenNodes, outputNodes } = getNodePositions();
      
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
          if ((i + j) % 3 === 0) { // Only draw some connections
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
      
      // Draw particles
      particles.forEach(particle => {
        if (particle.progress >= 0 && particle.progress <= 1) {
          const x = particle.x + (particle.targetX - particle.x) * particle.progress;
          const y = particle.y + (particle.targetY - particle.y) * particle.progress;
          
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(102, 126, 234, ${particle.opacity * (1 - particle.progress * 0.5)})`;
          ctx.fill();
          
          // Glow effect
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(102, 126, 234, ${particle.opacity * 0.3 * (1 - particle.progress * 0.5)})`;
          ctx.fill();
        }
      });
      
      // Update particles
      setParticles(prev => prev.map(p => ({
        ...p,
        progress: p.progress + 0.03,
      })).filter(p => p.progress <= 1.2));
      
      // Draw input layer nodes
      inputNodes.forEach((node) => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = animationPhase !== 'idle' ? activeNodeColor : nodeColor;
        ctx.fill();
      });
      
      // Draw hidden layer nodes
      hiddenNodes.forEach((node) => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = animationPhase === 'layer2' || animationPhase === 'complete' 
          ? activeNodeColor 
          : nodeColor;
        ctx.fill();
      });
      
      // Draw output layer nodes with confidence
      outputNodes.forEach((node, i) => {
        const conf = confidence[i] || 0;
        const isWinner = prediction === i;
        const radius = isWinner ? 10 : 6 + conf * 4;
        
        // Glow for winner
        if (isWinner && animationPhase === 'complete') {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 8, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
          ctx.fill();
        }
        
        // Node
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        
        if (isWinner && (animationPhase === 'complete' || animationPhase === 'idle')) {
          ctx.fillStyle = '#667eea';
        } else if (conf > 0.1) {
          ctx.fillStyle = `rgba(102, 126, 234, ${0.3 + conf * 0.7})`;
        } else {
          ctx.fillStyle = nodeColor;
        }
        ctx.fill();
        
        // Label
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
      
      ctx.fillText('Input', layerX[0], height - 10);
      ctx.fillText('784', layerX[0], height - 22);
      
      ctx.fillText('Hidden', layerX[1], height - 10);
      ctx.fillText('100', layerX[1], height - 22);
      
      ctx.fillText('Output', layerX[2], height - 10);
      ctx.fillText('10', layerX[2], height - 22);
      
      animationRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [particles, confidence, prediction, animationPhase, getNodePositions, height]);

  return (
    <div className="neural-net-viz">
      <h3>🔗 Network Architecture</h3>
      <canvas 
        ref={canvasRef} 
        width={width} 
        height={height}
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
