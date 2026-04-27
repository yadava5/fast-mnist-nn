import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080';

/**
 * Where a prediction was produced. The HTTP backend is preferred; the
 * browser WASM path is used whenever the server is unreachable or the
 * request explicitly forces offline mode.
 */
export type PredictionSource = 'server' | 'browser-wasm' | 'browser-js';

export interface PredictionResponse {
  prediction: number;
  confidence: number[];
  baseline_time_ms: number;
  optimized_time_ms: number;
  /**
   * Hidden-layer post-activation values (usually length 30 or 100, values in
   * [0, 1] after sigmoid/ReLU). Present when the C++ server exposes
   * activation telemetry; omitted by older builds.
   */
  hidden_activations?: number[];
  /**
   * Per-input-pixel gradient of the argmax logit wrt the 784-dim input.
   * Used for saliency overlays. Optional for older servers.
   */
  input_grad?: number[];
  /**
   * Where this prediction ran. Populated by predict(); the raw server
   * response does not carry this field, it is stamped client-side.
   */
  source?: PredictionSource;
}

export interface HealthResponse {
  status: string;
  version: string;
}

/**
 * Attempt the HTTP backend first. If the request fails for any reason
 * (network error, non-2xx, timeout, abort from a stale in-flight
 * request is re-raised), fall back to the browser WASM classifier so
 * the demo keeps working even when the server is offline.
 *
 * The TS wrapper is imported dynamically so the wasm glue chunk does
 * not land in the initial bundle -- it only downloads if/when the
 * fallback path triggers.
 */
export async function predict(pixels: number[], signal?: AbortSignal): Promise<PredictionResponse> {
  try {
    const response = await axios.post<PredictionResponse>(
      `${API_BASE}/predict`,
      { pixels },
      { signal, timeout: 2000 },
    );
    return { ...response.data, source: 'server' };
  } catch (serverErr) {
    // A user-initiated abort should not spill over into the WASM
    // path -- let the caller see the CanceledError like before.
    if (axios.isCancel(serverErr)) {
      throw serverErr;
    }
    if (signal?.aborted) {
      throw serverErr;
    }

    const { classifyInBrowser } = await import('../lib/wasmClassifier');
    const result = await classifyInBrowser(pixels);
    return { ...result, source: result.source ?? 'browser-wasm' };
  }
}

export async function healthCheck(): Promise<HealthResponse> {
  const response = await axios.get<HealthResponse>(`${API_BASE}/health`, {
    timeout: 1000,
  });
  return response.data;
}
