import { IconParallelFutures } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import { stageSignal, SIGNAL_BG_CLASS, SIGNAL_TEXT_CLASS } from "../../lib/severity"
import { STAGE_BENIGN } from "../../lib/types"
import type { MultipleFutures } from "../../lib/types"

/**
 * Phase 10 differentiator -- Multiple Parallel Futures. Built from a real
 * MC-Dropout ensemble (see futures/parallel_futures.py), not invented
 * percentages: each bar's width is the fraction of independently-sampled
 * rollouts that landed on that stage. Disclosed plainly rather than
 * presented as if it were literally branching attacker choices.
 */
export function ParallelFuturesPanel({ futures }: { futures: MultipleFutures }) {
  return (
    <Panel
      title="Multiple Parallel Futures"
      icon={<IconParallelFutures size={13} className="text-primary-container" />}
      aside={<span className="font-mono text-[10px] text-on-surface-variant">{futures.n_samples} samples</span>}
      className="reticle-corner"
      bodyClassName="flex flex-col gap-2.5 p-3"
    >
      {futures.outcomes.map((outcome) => {
        const signal = stageSignal({ stage: outcome.stage, basis: outcome.stage === STAGE_BENIGN ? "benign" : "attack_type" })
        return (
          <div key={outcome.stage} className="flex flex-col gap-1">
            <div className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-on-surface">{outcome.stage}</span>
              <DataReadout className={SIGNAL_TEXT_CLASS[signal]}>{(outcome.probability * 100).toFixed(0)}%</DataReadout>
            </div>
            <div className="h-1.5 w-full bg-surface-container">
              <div
                className={`h-1.5 ${SIGNAL_BG_CLASS[signal]}`}
                style={{ width: `${Math.max(2, outcome.probability * 100)}%` }}
              />
            </div>
          </div>
        )
      })}
      <p className="mt-1 font-mono text-[10px] leading-relaxed text-on-surface-variant opacity-70">
        MC-Dropout resampling of the same forecast context -- measures the model's own uncertainty, not independently observed alternate conditions.
      </p>
    </Panel>
  )
}
