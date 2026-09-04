import type { AnalyzeResponse } from "../types"
import sample from "./sample-response.json"

// Real captured Phase 7 API response (150K-row CIC-IDS2017 slice spanning
// the actual benign->DDoS transition, see project README's Phase 7
// section) -- used only for design-time component preview, never shipped
// as a fallback in the real app flow.
export const sampleResponse = sample as AnalyzeResponse
