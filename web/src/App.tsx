import { useCallback, useEffect, useRef, useState } from 'react';
import { DrawingCanvas } from './components/DrawingCanvas';
import { PredictionResult } from './components/PredictionResult';
import { NeuralNetViz } from './components/NeuralNetViz';
import { ThemeToggle } from './components/ThemeToggle';
import { NeuralNetHero } from './components/NeuralNetHero';
import { CommandPalette } from './components/CommandPalette';
import { predict, healthCheck } from './api/predict';
import { useTheme } from './hooks/useTheme';
import { useDebouncedCallback } from './hooks/useDebounce';
import './App.css';

function App() {
  // Hook mount applies theme + persists across toggles; ThemeToggle
  // reads its own useTheme() internally so we don't need the return here.
  useTheme();
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number[]>([]);
  const [baselineTime, setBaselineTime] = useState<number | null>(null);
  const [optimizedTime, setOptimizedTime] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const checkServer = async () => {
      try {
        await healthCheck();
        setServerStatus('online');
      } catch {
        setServerStatus('offline');
      }
    };
    checkServer();
    const interval = setInterval(checkServer, 5000);
    return () => clearInterval(interval);
  }, []);

  const performPrediction = useCallback(async (pixels: number[]) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsLoading(true);
    setIsAnimating(true);

    try {
      const result = await predict(pixels, abortControllerRef.current.signal);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
      setBaselineTime(result.baseline_time_ms);
      setOptimizedTime(result.optimized_time_ms);
    } catch (error) {
      if (error instanceof Error && error.name === 'CanceledError') {
        return;
      }
      console.error('Prediction failed:', error);
    } finally {
      setIsLoading(false);
      setTimeout(() => setIsAnimating(false), 300);
    }
  }, []);

  const { debouncedFn: handlePredict } = useDebouncedCallback(performPrediction, 300);

  return (
    <div className="app">
      <header className="header">
        <ThemeToggle />
        <h1>🧠 Fast MNIST Neural Network</h1>
        <p className="subtitle">
          Draw a digit and watch the neural network classify it in real-time
        </p>
        <div className={`server-status ${serverStatus}`}>
          <span className="status-dot"></span>
          {serverStatus === 'checking' && 'Connecting...'}
          {serverStatus === 'online' && 'Server Online'}
          {serverStatus === 'offline' && 'Server Offline - Start the backend'}
        </div>
        <p className="cmdk-hint" aria-hidden>
          <kbd>⌘</kbd>
          <kbd>K</kbd> for commands
        </p>
      </header>

      <section className="hero-section">
        <div className="hero-copy">
          <h2 className="hero-title">784 → 100 → 10</h2>
          <p className="hero-subtitle">
            A handwritten C++ multilayer perceptron with SIMD kernels and OpenMP, visualized.
          </p>
        </div>
        <div className="hero-visual">
          <NeuralNetHero />
        </div>
      </section>

      <main className="main-content">
        <div className="canvas-section">
          <h2>✏️ Draw Here</h2>
          <DrawingCanvas
            onPredict={handlePredict}
            disabled={serverStatus !== 'online'}
            isLoading={isLoading}
          />
        </div>

        <div className="result-section">
          <h2>🎯 Prediction</h2>
          <PredictionResult
            prediction={prediction}
            confidence={confidence}
            baselineTime={baselineTime}
            optimizedTime={optimizedTime}
            isLoading={isLoading}
          />
          <NeuralNetViz
            confidence={confidence}
            prediction={prediction}
            isAnimating={isAnimating}
          />
        </div>
      </main>

      <footer className="footer">
        <p>Built with C++ · SIMD kernels · OpenMP · Motion · React Three Fiber</p>
        <p className="author">By Ayush Yadav · Contributor: Shree Chaturvedi</p>
      </footer>

      <CommandPalette />
    </div>
  );
}

export default App;
