import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import { Badge } from "../primitives/Badge"
import { MomentumGauge } from "../primitives/MomentumGauge"
import { momentumSignal } from "../../lib/severity"
import type { MomentumScore } from "../../lib/types"

const DRIVER_LABELS: Record<string, string> = {
  probability_slope: "rising infiltration probability",
  newly_affected_host_ratio: "newly active hosts",
  stage_progression_rate: "frequent stage changes",
  forecast_trend: "a climbing forecast",
}

/**
 * Phase 9, Feature 1. Momentum measures rate of escalation, not standing
 * severity -- the mockup's fabricated "Edge Rate / Entropy / Escalation
 * Paths" 3-metric row is dropped (none of those are real backend fields);
 * what's real and kept: the score, its dominant driver, and the plain-
 * language explanation the backend already generates.
 */
export function MomentumPanel({ momentum }: { momentum: MomentumScore }) {
  const signal = momentumSignal(momentum.score)
  return (
    <Panel
      title="Cyber Threat Momentum"
      aside={<Badge signal={signal}>{signal === "critical" ? "critical" : signal === "warning" ? "escalating" : "stable"}</Badge>}
      className="reticle-corner"
      bodyClassName="flex flex-col items-center gap-2 p-4"
    >
      <MomentumGauge score={momentum.score} />
      <div className="-mt-2 flex items-baseline gap-1.5">
        <DataReadout className="text-[26px] font-bold">{momentum.score.toFixed(1)}</DataReadout>
        <span className="font-mono text-[11px] text-on-surface-variant">/ 100</span>
      </div>
      <div className="w-full border border-outline-variant bg-surface-container-lowest p-2.5 font-mono text-[10px]">
        <div className="mb-1 font-semibold uppercase tracking-wide text-on-surface-variant">primary driver</div>
        <div className="leading-relaxed text-on-surface">{DRIVER_LABELS[momentum.top_driver] ?? momentum.top_driver}</div>
      </div>
      {!momentum.attack_path_expansion_available && (
        <p className="text-center font-mono text-[9px] leading-relaxed text-on-surface-variant opacity-70">
          path-expansion signal not yet available -- Multiple Parallel Futures not built
        </p>
      )}
    </Panel>
  )
}
