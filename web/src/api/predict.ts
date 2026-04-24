import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080';

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
}

export interface HealthResponse {
  status: string;
  version: string;
}

export async function predict(pixels: number[], signal?: AbortSignal): Promise<PredictionResponse> {
  const response = await axios.post<PredictionResponse>(
    `${API_BASE}/predict`,
    { pixels },
    { signal },
  );
  return response.data;
}

export async function healthCheck(): Promise<HealthResponse> {
  const response = await axios.get<HealthResponse>(`${API_BASE}/health`);
  return response.data;
}
