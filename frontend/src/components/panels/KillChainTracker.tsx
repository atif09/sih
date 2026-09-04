import { IconKillChain } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { stageSignal, SIGNAL_BORDER_CLASS, SIGNAL_TEXT_CLASS } from "../../lib/severity"
import { MITRE_SIX_STAGES } from "../../lib/types"
import type { AnalyzeResponse } from "../../lib/types"

type StageStatus = "observed" | "current" | "forecast" | "none"

interface StageCell {
  stage: string
  status: StageStatus
  detail: string
  confidence: number | null
  basis: string | null
}

/**
 * The Stitch mockup's 6-step kill chain used fabricated per-stage MITRE
 * technique IDs and countdown timers (T1595, T1021.002, "IN 3m 42s"...) --
 * none of that exists in this model's real output. What IS real: the
 * model's own stage vocabulary is exactly these 6 labels (5 canonical +
 * rollout/mitre.py's honest 6th "Denial of Service / Impact" for traffic
 * that doesn't fit the kill chain), and the K-step rollout already tells us
 * which stage comes next AND which stages the forecast passes through
 * later -- so each cell below is classified from data the backend already
 * returns, not invented.
 */
function buildKillChain(result: AnalyzeResponse): StageCell[] {
  const observedCounts = new Map<string, number>()
  for (const t of result.timeline) {
    observedCounts.set(t.mitre_stage.stage, (observedCounts.get(t.mitre_stage.stage) ?? 0) + 1)
  }

  const current = result.forecast[0]?.mitre_stage
  const laterForecast = result.forecast.slice(1)

  return MITRE_SIX_STAGES.map((stage): StageCell => {
    if (current && current.stage === stage) {
      return {
        stage,
        status: "current",
        detail: `${current.basis === "attack_type" ? "model-predicted" : current.basis === "heuristic_fallback" ? "heuristic fallback" : "benign"} · ${(current.confidence * 100).toFixed(0)}% confidence`,
        confidence: current.confidence,
        basis: current.basis,
      }
    }
    const idx = laterForecast.findIndex((f) => f.mitre_stage.stage === stage)
    if (idx !== -1) {
      const step = idx + 2 // +1 for 0-index, +1 because laterForecast skips forecast[0]
      const minutesAhead = (step * result.window_size_seconds) / 60
      return {
        stage,
        status: "forecast",
        detail: `forecast step ${step} · ~T+${minutesAhead.toFixed(1)}m`,
        confidence: laterForecast[idx].mitre_stage.confidence,
        basis: laterForecast[idx].mitre_stage.basis,
      }
    }
    const count = observedCounts.get(stage)
    if (count) {
      return { stage, status: "observed", detail: `${count} observed window${count === 1 ? "" : "s"}`, confidence: null, basis: null }
    }
    return { stage, status: "none", detail: "not indicated", confidence: null, basis: null }
  })
}

export function KillChainTracker({ result }: { result: AnalyzeResponse }) {
  const cells = buildKillChain(result)

  return (
    <Panel
      title="MITRE ATT&CK Kill-Chain Forecast"
      icon={<IconKillChain size={13} className="text-primary-container" />}
      className="reticle-corner"
      bodyClassName="grid grid-cols-2 gap-2 p-3 md:grid-cols-6"
    >
      {cells.map((cell, i) => {
        const signal = cell.status === "current" ? stageSignal({ stage: cell.stage, basis: cell.basis ?? "" }) : null
        const isActive = cell.status === "current"
        const isForecast = cell.status === "forecast"
        const isObserved = cell.status === "observed"

        return (
          <div
            key={cell.stage}
            className={`flex flex-col justify-between border p-2.5 ${
              isActive
                ? `border-2 ${signal ? SIGNAL_BORDER_CLASS[signal] : "border-error"} bg-surface-container-high`
                : isForecast
                  ? "border-dashed border-warning/60 bg-surface-container-lowest"
                  : isObserved
                    ? "border-outline-variant bg-surface-container-lowest opacity-90"
                    : "border-dashed border-outline-variant/60 bg-surface-container-lowest opacity-50"
            }`}
          >
            <div className="flex items-center justify-between font-mono text-[9px]">
              <span className="text-on-surface-variant">
                0{i + 1}. {cell.stage.toUpperCase()}
              </span>
              <span
                className={
                  isActive
                    ? `font-bold ${signal ? SIGNAL_TEXT_CLASS[signal] : "text-error"}`
                    : isForecast
                      ? "text-warning"
                      : isObserved
                        ? "text-secondary-fixed-dim"
                        : "text-outline"
                }
              >
                {isActive ? "[CURRENT]" : isForecast ? "[FORECAST]" : isObserved ? "[OBSERVED]" : "[--]"}
              </span>
            </div>
            <div className="mt-1.5 font-mono text-[10px] leading-snug text-on-surface">{cell.detail}</div>
          </div>
        )
      })}
    </Panel>
  )
}
