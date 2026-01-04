import axios from 'axios';

const API_BASE = 'http://localhost:8080';

export interface PredictionResponse {
  prediction: number;
  confidence: number[];
  baseline_time_ms: number;
  optimized_time_ms: number;
}

export interface HealthResponse {
  status: string;
  version: string;
}

export async function predict(
  pixels: number[],
  signal?: AbortSignal
): Promise<PredictionResponse> {
  const response = await axios.post<PredictionResponse>(
    `${API_BASE}/predict`,
    { pixels },
    { signal }
  );
  return response.data;
}

export async function healthCheck(): Promise<HealthResponse> {
  const response = await axios.get<HealthResponse>(`${API_BASE}/health`);
  return response.data;
}

