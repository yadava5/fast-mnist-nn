import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, MotionConfig, motion } from 'motion/react';
import { BrainCircuit, PencilLine, Target } from 'lucide-react';
import { DrawingCanvas } from './components/DrawingCanvas';
import { PredictionResult } from './components/PredictionResult';
import { ActivationPanels } from './components/ActivationPanels';
import { ThemeToggle } from './components/ThemeToggle';
import { NeuralNetHero } from './components/NeuralNetHero';
import { HeroBackdrop } from './components/HeroBackdrop';
import { PipelineShowcase } from './components/PipelineShowcase';
import { CommandPalette } from './components/CommandPalette';
import { ScrollProgress } from './components/ScrollProgress';
import { createSampleDigitFive } from './components/sampleDigits';
import type { Stroke } from './components/strokeReducer';
import { predict, healthCheck, type PredictionSource } from './api/predict';
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
  const [hiddenActivations, setHiddenActivations] = useState<number[] | undefined>();
  const [inputGrad, setInputGrad] = useState<number[] | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [predictionSource, setPredictionSource] = useState<PredictionSource | null>(null);
  const [strokeCount, setStrokeCount] = useState(0);
  const [clearSignal, setClearSignal] = useState(0);
  const [sampleSignal, setSampleSignal] = useState(0);
  const [sampleStrokes, setSampleStrokes] = useState<Stroke[] | null>(() =>
    createSampleDigitFive(),
  );

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
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);

    try {
      const result = await predict(pixels, controller.signal);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
      setBaselineTime(result.baseline_time_ms > 0 ? result.baseline_time_ms : null);
      setOptimizedTime(result.optimized_time_ms);
      setHiddenActivations(result.hidden_activations);
      setInputGrad(result.input_grad);
      setPredictionSource(result.source ?? null);
    } catch (error) {
      if (error instanceof Error && error.name === 'CanceledError') {
        return;
      }
      console.error('Prediction failed:', error);
    } finally {
      if (abortControllerRef.current === controller) {
        setIsLoading(false);
      }
    }
  }, []);

  const { debouncedFn: handlePredict, cancel: cancelPredictionDebounce } = useDebouncedCallback(
    performPrediction,
    300,
  );

  const resetPrediction = useCallback(() => {
    cancelPredictionDebounce();
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setPrediction(null);
    setConfidence([]);
    setBaselineTime(null);
    setOptimizedTime(null);
    setHiddenActivations(undefined);
    setInputGrad(undefined);
    setPredictionSource(null);
    setIsLoading(false);
  }, [cancelPredictionDebounce]);
  const canClear = strokeCount > 0 || prediction !== null || confidence.length > 0 || isLoading;

  const handleClearCanvas = useCallback(() => {
    resetPrediction();
    setClearSignal((signal) => signal + 1);
  }, [resetPrediction]);

  const handleLoadSampleDigit = useCallback(() => {
    resetPrediction();
    setSampleStrokes(createSampleDigitFive());
    setSampleSignal((signal) => signal + 1);
  }, [resetPrediction]);

  const sourceLabel =
    predictionSource === 'browser-wasm'
      ? 'wasm-mode'
      : predictionSource === 'browser-js'
        ? 'js-demo-mode'
        : null;

  return (
    <MotionConfig transition={{ type: 'spring', stiffness: 260, damping: 28, mass: 0.9 }}>
      <div className="app">
        <ScrollProgress />
        <motion.header className="header" initial={false}>
          <ThemeToggle className="theme-toggle" />
          <h1 className="brand-title">
            <BrainCircuit className="brand-mark" size={34} strokeWidth={1.75} aria-hidden />
            <span>Fast MNIST Neural Network</span>
          </h1>
          <p className="subtitle">
            Draw a digit and watch the neural network classify it in real-time
          </p>
          <div className={`server-status ${serverStatus}`}>
            <span className="status-dot"></span>
            {serverStatus === 'checking' && 'Connecting...'}
            {serverStatus === 'online' && 'Server Online'}
            {serverStatus === 'offline' &&
              predictionSource !== 'browser-wasm' &&
              predictionSource !== 'browser-js' &&
              'Server Offline - browser fallback ready'}
            {serverStatus === 'offline' &&
              predictionSource === 'browser-wasm' &&
              'Running in browser (WASM)'}
            {serverStatus === 'offline' &&
              predictionSource === 'browser-js' &&
              'Running in browser (JS fallback)'}
          </div>
          <AnimatePresence mode="wait">
            {sourceLabel && (
              <motion.div
                key={sourceLabel}
                className={`runtime-badge ${predictionSource ?? ''}`}
                aria-label={`Prediction source: ${sourceLabel}`}
                initial={{ opacity: 0, y: -6, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.96 }}
                transition={{ duration: 0.18 }}
              >
                {sourceLabel}
              </motion.div>
            )}
          </AnimatePresence>
          <p className="cmdk-hint" aria-hidden>
            <kbd>⌘</kbd>
            <kbd>K</kbd> for commands
          </p>
        </motion.header>

        <main id="draw" className="main-content">
          <motion.div className="canvas-section" initial={false}>
            <h2 className="section-heading">
              <PencilLine size={20} strokeWidth={1.8} aria-hidden />
              <span>Draw Here</span>
            </h2>
            <DrawingCanvas
              onPredict={handlePredict}
              onClear={resetPrediction}
              onStrokeCountChange={setStrokeCount}
              clearSignal={clearSignal}
              sampleSignal={sampleSignal}
              sampleStrokes={sampleStrokes}
              /*
               * Allow drawing while the server is still being polled
               * (the request will either succeed or fall through to the
               * WASM classifier) and when the server is offline (the
               * fallback handles the request). Only 'checking' locks the
               * canvas, briefly, during the initial health probe.
               */
              disabled={serverStatus === 'checking'}
              isLoading={isLoading}
            />
          </motion.div>

          <motion.div id="results" className="result-section" initial={false}>
            <h2 className="section-heading">
              <Target size={20} strokeWidth={1.8} aria-hidden />
              <span>Prediction</span>
            </h2>
            <PredictionResult
              prediction={prediction}
              confidence={confidence}
              baselineTime={baselineTime}
              optimizedTime={optimizedTime}
              isLoading={isLoading}
            />
            <ActivationPanels
              prediction={prediction}
              confidence={confidence}
              hiddenActivations={hiddenActivations}
              inputGrad={inputGrad}
            />
          </motion.div>
        </main>

        <motion.section id="network" className="hero-section" initial={false}>
          <HeroBackdrop />
          <div className="hero-inner">
            <div className="hero-copy">
              <h2 className="hero-title">784 → 100 → 10</h2>
              <p className="hero-subtitle">
                A handwritten C++ multilayer perceptron with SIMD kernels and OpenMP, visualized.
              </p>
            </div>
            <div className="hero-visual">
              <NeuralNetHero />
            </div>
          </div>
        </motion.section>

        <PipelineShowcase />

        <footer className="footer">
          <p>Built with C++ · SIMD kernels · OpenMP · Motion · React Three Fiber</p>
          <p className="author">By Ayush Yadav · Contributor: Shree Chaturvedi</p>
        </footer>

        <CommandPalette
          onClearCanvas={handleClearCanvas}
          onLoadSampleDigit={handleLoadSampleDigit}
          canClear={canClear}
        />
      </div>
    </MotionConfig>
  );
}

export default App;
