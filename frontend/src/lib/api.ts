import axios from "axios"
import type { AnalyzeResponse, CalibrationResponse, PlanResponseRequest, PlanResponseResponse, ROCPRCurvesResponse, SimulateRequest, SimulateResponse } from "./types"

// Vite dev server proxies /api -> http://localhost:8000 (see vite.config.ts).
// No external hosts anywhere -- fully offline against the local backend.
const client = axios.create({ baseURL: "/api" })

export async function analyze(
  file: File,
  kind: "csv" | "pcap",
  k: number,
): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append(kind === "csv" ? "csv_file" : "pcap_file", file)
  const res = await client.post<AnalyzeResponse>("/analyze", form, {
    params: { k },
    headers: { "Content-Type": "multipart/form-data" },
  })
  return res.data
}

// Phase 9, Feature 3. Acts on the context cached server-side from the most
// recent /analyze call -- there is no file upload here.
export async function simulate(request: SimulateRequest): Promise<SimulateResponse> {
  const res = await client.post<SimulateResponse>("/simulate", request)
  return res.data
}

// Response Planner (differentiator). Same cached-context convention as
// simulate() above -- call analyze() first.
export async function planResponse(request: PlanResponseRequest): Promise<PlanResponseResponse> {
  const res = await client.post<PlanResponseResponse>("/plan-response", request)
  return res.data
}

export interface HealthResponse {
  status: string
  device: string | null
  window_size_seconds: number | null
  seq_len: number | null
}

export interface ModelInfoResponse {
  temporal_type: string
  latent_dim: number
  attack_type_vocab: string[]
  window_size_seconds: number
  seq_len: number
  default_k: number
  n_parameters: number
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await client.get<HealthResponse>("/health")
  return res.data
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await client.get<ModelInfoResponse>("/model-info")
  return res.data
}

// Fixed model-level ROC/PR curve data (held-out day), independent of any
// uploaded capture -- see benchmark/run_benchmark.py and ModelPerformance.
export async function getROCPRCurves(): Promise<ROCPRCurvesResponse> {
  const res = await client.get<ROCPRCurvesResponse>("/model-performance/roc-pr")
  return res.data
}

// Fixed model-level reliability diagram (held-out day, world model only).
export async function getCalibration(): Promise<CalibrationResponse> {
  const res = await client.get<CalibrationResponse>("/model-performance/calibration")
  return res.data
}
