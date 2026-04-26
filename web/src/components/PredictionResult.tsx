import { AnimatePresence, motion } from 'motion/react';

interface PredictionResultProps {
  prediction: number | null;
  confidence: number[];
  baselineTime: number | null;
  optimizedTime: number | null;
  isLoading: boolean;
}

export function PredictionResult({
  prediction,
  confidence,
  baselineTime,
  optimizedTime,
  isLoading,
}: PredictionResultProps) {
  const speedup = baselineTime && optimizedTime ? (baselineTime / optimizedTime).toFixed(1) : null;

  return (
    <div className="prediction-result">
      <div className="prediction-main">
        <AnimatePresence mode="wait">
          {isLoading ? (
            <motion.div
              key="loading"
              className="loading"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.16 }}
            >
              <div className="spinner"></div>
              <span>Analyzing...</span>
            </motion.div>
          ) : prediction !== null ? (
            <motion.div
              key={`prediction-${prediction}`}
              initial={{ opacity: 0, scale: 0.92 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ type: 'spring', stiffness: 380, damping: 26 }}
            >
              <div className="predicted-digit">{prediction}</div>
              <div className="confidence-label">
                {((confidence[prediction] ?? 0) * 100).toFixed(1)}% confidence
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="placeholder"
              className="placeholder"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.16 }}
            >
              <span className="placeholder-icon">✏️</span>
              <span>Draw a digit (0-9)</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {(baselineTime !== null || optimizedTime !== null) && (
        <div className="timing-comparison">
          <h3>⚡ Performance Comparison</h3>
          <div className="timing-bars">
            <div className="timing-row">
              <span className="timing-label">Baseline:</span>
              <div className="timing-bar-container">
                <div
                  className="timing-bar baseline"
                  style={{
                    width: baselineTime ? `${Math.min(100, baselineTime / 10)}%` : '0%',
                  }}
                />
              </div>
              <span className="timing-value">
                {baselineTime !== null ? `${baselineTime.toFixed(2)}ms` : '-'}
              </span>
            </div>
            <div className="timing-row">
              <span className="timing-label">Optimized:</span>
              <div className="timing-bar-container">
                <div
                  className="timing-bar optimized"
                  style={{
                    width: optimizedTime ? `${Math.min(100, optimizedTime / 10)}%` : '0%',
                  }}
                />
              </div>
              <span className="timing-value">
                {optimizedTime !== null ? `${optimizedTime.toFixed(2)}ms` : '-'}
              </span>
            </div>
          </div>
          {speedup && parseFloat(speedup) > 1 && (
            <div className="speedup-badge">🚀 {speedup}x faster!</div>
          )}
        </div>
      )}

      {confidence.length > 0 && (
        <div className="confidence-chart">
          <h3>📊 Confidence Scores</h3>
          <div className="confidence-bars">
            {confidence.map((conf, digit) => (
              <div key={digit} className="confidence-row">
                <span className="digit-label">{digit}</span>
                <div className="confidence-bar-container">
                  <div
                    className={`confidence-bar ${digit === prediction ? 'active' : ''}`}
                    style={{ width: `${conf * 100}%` }}
                  />
                </div>
                <span className="confidence-value">{(conf * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
