import { IconReset } from "../../lib/icons"
import { UploadPanel } from "../panels/UploadPanel"
import type { AnalyzeResponse } from "../../lib/types"

export type Mode = "live" | "response" | "replay"

const MODES: { id: Mode; label: string }[] = [
  { id: "live", label: "Mode 1: Live Feed" },
  { id: "response", label: "Mode 2: Response Center" },
  { id: "replay", label: "Mode 3: Attack Replay" },
]

/**
 * Top cockpit header, adapted from the Stitch mockup. Dropped from the
 * source design (no real backend behind them): the mission clock, the
 * audio-alarm toggle, "system override," and the model checkpoint/loss/
 * accuracy readout -- this API doesn't expose training metrics or raw
 * alerting state. Kept and wired to real data: file/window/context counts
 * (only once a capture has been analyzed), the mode switch, and the upload
 * control itself.
 */
export function Header({
  mode,
  onModeChange,
  result,
  filename,
  k,
  onKChange,
  onResult,
  onReset,
  onHome,
}: {
  mode: Mode
  onModeChange: (m: Mode) => void
  result: AnalyzeResponse | null
  filename: string
  k: number
  onKChange: (k: number) => void
  onResult: (r: AnalyzeResponse, filename: string) => void
  onReset: () => void
  onHome?: () => void
}) {
  return (
    <header className="sticky top-0 z-50 flex w-full flex-col border-b border-outline-variant bg-surface-container-lowest">
      <div className="flex items-center justify-between border-b border-outline-variant/60 px-5 py-2.5">
        <div className="flex items-center gap-3">
          <button
            onClick={onHome}
            disabled={!onHome}
            className="flex items-center gap-2 font-sans text-sm font-semibold tracking-tight text-on-surface disabled:cursor-default"
          >
            <span className="h-2 w-2 rounded-full bg-primary-container" />
            NeuralShield // GNN Forecasting
          </button>
        </div>

        <div className="flex items-center gap-3">
          <UploadPanel onResult={onResult} k={k} onKChange={onKChange} />
          {result && (
            <button
              onClick={onReset}
              title="Clear the current analysis"
              className="flex items-center gap-1.5 border border-outline-variant bg-surface-container px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider text-on-surface-variant transition-colors hover:text-error"
            >
              <IconReset size={12} />
              reset
            </button>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 px-5 py-2">
        <div className="flex items-center gap-1">
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => onModeChange(m.id)}
              className={`flex items-center gap-1.5 border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors ${
                mode === m.id
                  ? "border-outline-variant bg-surface-container text-on-surface"
                  : "border-transparent text-on-surface-variant hover:text-on-surface"
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${mode === m.id ? "bg-primary-container" : "border border-on-surface-variant"}`} />
              {m.label}
            </button>
          ))}
        </div>

        {result && (
          <div className="flex items-center gap-3 font-mono text-[11px] text-on-surface-variant">
            <span>
              ingest <span className="text-on-surface">{filename}</span>
            </span>
            <span className="text-outline-variant">/</span>
            <span>
              windows <span className="text-on-surface">{result.num_windows}</span>
            </span>
            <span className="text-outline-variant">/</span>
            <span>
              context <span className="text-on-surface">{result.context_len}</span>
            </span>
            <span className="text-outline-variant">/</span>
            <span>
              horizon <span className="text-on-surface">+{result.forecast.length} steps</span>
            </span>
          </div>
        )}
      </div>
    </header>
  )
}
