import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import { Badge } from "../primitives/Badge"
import { MomentumGauge } from "../primitives/MomentumGauge"
import { momentumSignal, probabilitySignal } from "../../lib/severity"
import type { MomentumScore } from "../../lib/types"

/**
 * Phase 9, Feature 1. Momentum measures rate of escalation, not standing
 * severity -- the mockup's fabricated "Edge Rate / Entropy / Escalation
 * Paths" 3-metric row is dropped (none of those are real backend fields);
 * what's real and kept: the score and the real plain-language explanation
 * the backend generates (momentum/score.py's `_build_explanation` --
 * e.g. "Momentum driven primarily by rising infiltration probability
 * (+0.0123/window)"), shown verbatim rather than mapped through a
 * separate hardcoded label that would risk drifting out of sync with it.
 *
 * `currentProbability` pairs the CURRENT point-in-time severity right next
 * to momentum's rate-of-change -- the whole point of this feature is "how
 * fast," not "how bad," and that only reads clearly next to the "how bad"
 * number it's rate-of-change of.
 */
export function MomentumPanel({ momentum, currentProbability }: { momentum: MomentumScore; currentProbability?: number }) {
  const signal = momentumSignal(momentum.score)
  return (
    <Panel
      title="Cyber Threat Momentum"
      aside={<Badge signal={signal}>{signal === "critical" ? "critical" : signal === "warning" ? "escalating" : "stable"}</Badge>}
      className="reticle-corner"
      bodyClassName="flex flex-col items-center gap-2 p-4"
    >
      <MomentumGauge score={momentum.score} />

      <div className="flex w-full items-stretch divide-x divide-outline-variant border border-outline-variant">
        <div className="flex flex-1 flex-col items-center gap-0.5 py-2">
          <span className="font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">momentum (rate)</span>
          <div className="flex items-baseline gap-1">
            <DataReadout className="text-[22px] font-bold">{momentum.score.toFixed(1)}</DataReadout>
            <span className="font-mono text-[10px] text-on-surface-variant">/100</span>
          </div>
        </div>
        {currentProbability !== undefined && (
          <div className="flex flex-1 flex-col items-center gap-0.5 py-2">
            <span className="font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">current risk (severity)</span>
            <DataReadout className={`text-[22px] font-bold ${probabilitySignal(currentProbability) === "critical" ? "text-error" : probabilitySignal(currentProbability) === "warning" ? "text-warning" : ""}`}>
              {currentProbability.toFixed(3)}
            </DataReadout>
          </div>
        )}
      </div>

      <div className="w-full border border-outline-variant bg-surface-container-lowest p-2.5 font-mono text-[10px]">
        <div className="mb-1 font-semibold uppercase tracking-wide text-on-surface-variant">why</div>
        <div className="leading-relaxed text-on-surface">{momentum.explanation}</div>
      </div>
      {!momentum.attack_path_expansion_available && (
        <p className="text-center font-mono text-[9px] leading-relaxed text-on-surface-variant opacity-70">
          path-expansion signal not yet available -- Multiple Parallel Futures not built
        </p>
      )}
    </Panel>
  )
}
