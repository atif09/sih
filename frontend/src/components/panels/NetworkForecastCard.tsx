import { IconArrowDown, IconArrowUp, IconForecastView } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import type { NetworkForecastView } from "../../lib/types"

const STAT_LABELS: Record<string, string> = {
  flow_count: "flow count",
  max_dst_ports_touched: "most-scanned host's dst ports",
  bytes_out: "bytes out",
  bytes_in: "bytes in",
}

function formatValue(name: string, value: number): string {
  if (name === "bytes_out" || name === "bytes_in") {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)} MB`
    if (value >= 1_000) return `${(value / 1_000).toFixed(1)} KB`
    return `${value.toFixed(0)} B`
  }
  return value.toFixed(name === "max_dst_ports_touched" ? 1 : 0)
}

function reliabilityLabel(r2: number | undefined): { label: string; className: string } {
  if (r2 === undefined) return { label: "unknown", className: "text-on-surface-variant" }
  if (r2 >= 0.4) return { label: "moderate signal", className: "text-secondary-fixed-dim" }
  if (r2 >= 0.2) return { label: "weak signal", className: "text-warning" }
  return { label: "low confidence", className: "text-error" }
}

/**
 * Network Forecast View (differentiator): "now vs +K minutes" comparison
 * card -- see models/forecast_decoder.py. now_stats are real ground truth;
 * predicted_stats come from a small decoder trained on top of the frozen
 * world model, NOT the base checkpoint's own training. Real held-out R^2
 * per stat is always shown (0.28-0.49 on this project's own held-out day)
 * so this reads as "moderate, disclosed signal," not a confident claim.
 */
export function NetworkForecastCard({ forecast }: { forecast: NetworkForecastView | null }) {
  if (!forecast) {
    return (
      <Panel title="Network Forecast View" icon={<IconForecastView size={13} className="text-primary-container" />} className="reticle-corner flex flex-1 items-center justify-center">
        <p className="p-4 text-center font-mono text-[11px] text-on-surface-variant">no forecast decoder shipped with this checkpoint -- run scripts/train_forecast_decoder.py</p>
      </Panel>
    )
  }

  const stats = Object.keys(forecast.now_stats)

  return (
    <Panel
      title="Network Forecast View"
      icon={<IconForecastView size={13} className="text-primary-container" />}
      aside={<span className="font-mono text-[10px] text-primary-container">now vs. step +{forecast.step}</span>}
      className="reticle-corner flex flex-col"
      bodyClassName="flex flex-col gap-2 p-3"
    >
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {stats.map((name) => {
          const now = forecast.now_stats[name]
          const predicted = forecast.predicted_stats[name]
          const delta = predicted - now
          const pctChange = now !== 0 ? (delta / Math.abs(now)) * 100 : null
          const r2 = forecast.held_out_r2?.[name]
          const reliability = reliabilityLabel(r2)
          const rising = delta > 0

          return (
            <div key={name} className="flex flex-col gap-1 border border-outline-variant p-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">{STAT_LABELS[name] ?? name}</span>
                <span className={`font-mono text-[9px] ${reliability.className}`}>{reliability.label}{r2 !== undefined ? ` (R² ${r2.toFixed(2)})` : ""}</span>
              </div>
              <div className="flex items-center gap-2 font-mono text-[13px] font-bold text-on-surface">
                <span>{formatValue(name, now)}</span>
                <span className={`flex items-center gap-0.5 ${rising ? "text-error" : "text-secondary-fixed-dim"}`}>
                  {rising ? <IconArrowUp size={11} /> : <IconArrowDown size={11} />}
                  {formatValue(name, predicted)}
                  {pctChange !== null && <span className="text-[10px] font-normal text-on-surface-variant">({pctChange >= 0 ? "+" : ""}{pctChange.toFixed(0)}%)</span>}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </Panel>
  )
}
