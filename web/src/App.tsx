import { useState, useEffect } from 'react';
import { DrawingCanvas } from './components/DrawingCanvas';
import { PredictionResult } from './components/PredictionResult';
import { predict, healthCheck } from './api/predict';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number[]>([]);
  const [baselineTime, setBaselineTime] = useState<number | null>(null);
  const [optimizedTime, setOptimizedTime] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');

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

  const handlePredict = async (pixels: number[]) => {
    setIsLoading(true);
    try {
      const result = await predict(pixels);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
      setBaselineTime(result.baseline_time_ms);
      setOptimizedTime(result.optimized_time_ms);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Failed to get prediction. Is the server running?');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🧠 Fast MNIST Neural Network</h1>
        <p className="subtitle">
          Draw a digit and see the neural network classify it in real-time
        </p>
        <div className={`server-status ${serverStatus}`}>
          <span className="status-dot"></span>
          {serverStatus === 'checking' && 'Connecting...'}
          {serverStatus === 'online' && 'Server Online'}
          {serverStatus === 'offline' && 'Server Offline - Start the backend'}
        </div>
      </header>

      <main className="main-content">
        <div className="canvas-section">
          <h2>✏️ Draw Here</h2>
          <DrawingCanvas 
            onPredict={handlePredict} 
            disabled={isLoading || serverStatus !== 'online'} 
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
        </div>
      </main>

      <footer className="footer">
        <p>
          Built with C++ • SIMD Optimizations • OpenMP Parallelization
        </p>
        <p className="author">
          By Ayush Yadav • Contributor: Shree Chaturvedi
        </p>
      </footer>
    </div>
  );
}

export default App;
