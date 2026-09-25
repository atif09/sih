import { useState } from "react"
import { IconParallelFutures } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import { Badge } from "../primitives/Badge"
import { stageSignal, SIGNAL_HEX } from "../../lib/severity"
import { MITRE_SIX_STAGES, STAGE_BENIGN } from "../../lib/types"
import type { MultipleFutures } from "../../lib/types"

const ROOT_X = 70
const LEAF_X = 340
const VIEWBOX_WIDTH = 640
const ROW_HEIGHT = 46
const TOP_PAD = 30

// Real MITRE sequence position (01-06), same numbering convention as
// KillChainTracker -- this app's real vocabulary is 6 stages, not the 5
// the differentiator brief's own framing assumes.
function stageNumber(stage: string): string {
  const i = MITRE_SIX_STAGES.indexOf(stage as (typeof MITRE_SIX_STAGES)[number])
  return i === -1 ? "--" : String(i + 1).padStart(2, "0")
}

/**
 * Multiple Parallel Futures as a branching diagram -- current deterministic
 * forecast (root, left) fanning out into each MC-Dropout outcome (leaves,
 * right), branch thickness/opacity scaled by real probability.
 *
 * Deliberately 2 levels, not deeper: futures/parallel_futures.py only
 * tallies each of the 40 independent rollouts' FINAL-step stage -- the
 * intermediate path each sample took through steps 1..k-1 is never
 * computed or returned anywhere, so a deeper tree would mean fabricating
 * branch structure that doesn't exist. This shows exactly the real shape
 * of the data: one branch point, N real outcomes.
 *
 * Root is styled with the app's "observed" solid treatment (per the
 * design brief) even though it's technically also a forecast (the single
 * no-dropout/eval-mode rollout) -- it's the fixed anchor the stochastic
 * branches diverge from, labeled precisely as "deterministic forecast"
 * rather than implying it's literally past-observed data.
 */
export function ParallelFuturesTree({ futures }: { futures: MultipleFutures }) {
  const [selected, setSelected] = useState(0)
  const outcomes = futures.outcomes
  const active = outcomes[selected]

  const height = TOP_PAD * 2 + Math.max(0, outcomes.length - 1) * ROW_HEIGHT
  const rootY = height / 2

  return (
    <Panel
      title="Multiple Parallel Futures"
      icon={<IconParallelFutures size={13} className="text-primary-container" />}
      aside={<span className="font-mono text-[10px] text-on-surface-variant">{futures.n_samples} MC-Dropout samples</span>}
      className="reticle-corner"
      bodyClassName="flex flex-col gap-0 lg:flex-row"
    >
      <div className="flex-1 overflow-x-auto p-3">
        <svg viewBox={`0 0 ${VIEWBOX_WIDTH} ${height}`} width="100%" height={height} role="img" aria-label="branching diagram of predicted future attack stages">
          {outcomes.map((o, i) => {
            const y = TOP_PAD + i * ROW_HEIGHT
            const isSel = i === selected
            const strokeWidth = 2 + o.probability * 14
            const opacity = Math.max(0.28, o.probability)
            const midX = (ROOT_X + LEAF_X) / 2
            const path = `M ${ROOT_X} ${rootY} C ${midX} ${rootY}, ${midX} ${y}, ${LEAF_X - 8} ${y}`
            return (
              <g key={o.stage} className="cursor-pointer" onClick={() => setSelected(i)} onMouseEnter={() => setSelected(i)}>
                {/* wide invisible hit-area -- the real stroke can be as thin as 2px at low probability, too small a hover/click target on its own */}
                <path d={path} stroke="transparent" strokeWidth={18} fill="none" />
                <path
                  d={path}
                  stroke="#ffb4ab"
                  strokeWidth={strokeWidth}
                  strokeDasharray={isSel ? undefined : "6 4"}
                  fill="none"
                  opacity={isSel ? 1 : opacity * 0.85}
                  style={{ transition: "opacity 150ms, stroke-width 150ms" }}
                />
              </g>
            )
          })}

          {/* Root -- deterministic (no-dropout) forecast, "observed"-style solid
              treatment. Shows the stage NUMBER, not the full name (the box is
              too narrow for long names like "Denial of Service / Impact" to
              ever fit reliably) -- a native <title> carries the full name for
              hover/accessibility, same info the leaves spell out in full. */}
          <g>
            <rect x={8} y={rootY - 20} width={ROOT_X + 24} height={40} fill="#09090b" />
            <title>{`Current deterministic forecast: ${futures.deterministic_stage}`}</title>
            <text x={8 + (ROOT_X + 24) / 2} y={rootY - 6} textAnchor="middle" fontFamily="JetBrains Mono" fontSize={7.5} fill="#ffffff" opacity={0.7}>
              CURRENT
            </text>
            <text x={8 + (ROOT_X + 24) / 2} y={rootY + 12} textAnchor="middle" fontFamily="JetBrains Mono" fontSize={16} fontWeight={700} fill="#ffffff">
              {stageNumber(futures.deterministic_stage)}
            </text>
          </g>

          {/* Leaves */}
          {outcomes.map((o, i) => {
            const y = TOP_PAD + i * ROW_HEIGHT
            const isSel = i === selected
            const signal = stageSignal({ stage: o.stage, basis: o.stage === STAGE_BENIGN ? "benign" : "attack_type" })
            return (
              <g key={o.stage} className="cursor-pointer" onClick={() => setSelected(i)} onMouseEnter={() => setSelected(i)}>
                <circle cx={LEAF_X} cy={y} r={isSel ? 5.5 : 4} fill={SIGNAL_HEX[signal]} stroke={isSel ? "#09090b" : "none"} strokeWidth={1.5} style={{ transition: "r 150ms" }} />
                <text x={LEAF_X + 12} y={y - 5} fontFamily="Geist Sans, sans-serif" fontSize={10.5} fontWeight={isSel ? 700 : 500} fill="#09090b">
                  {stageNumber(o.stage)}. {o.stage}
                </text>
                <text x={LEAF_X + 12} y={y + 9} fontFamily="JetBrains Mono" fontSize={9} fill="#71717a">
                  {(o.probability * 100).toFixed(0)}% · n={o.n_samples}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      {active && (
        <div className="flex w-full shrink-0 flex-col gap-2.5 border-t border-outline-variant p-3 lg:w-64 lg:border-l lg:border-t-0">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">selected outcome</span>
            <Badge signal={stageSignal({ stage: active.stage, basis: active.stage === STAGE_BENIGN ? "benign" : "attack_type" })}>{active.stage}</Badge>
          </div>
          <div className="grid grid-cols-2 gap-2 font-mono text-[10px]">
            <div>
              <div className="text-on-surface-variant">probability</div>
              <DataReadout className="text-lg font-bold">{(active.probability * 100).toFixed(1)}%</DataReadout>
            </div>
            <div>
              <div className="text-on-surface-variant">mean infiltration p</div>
              <DataReadout className="text-lg font-bold">{active.mean_infiltration_probability.toFixed(3)}</DataReadout>
            </div>
          </div>
          <div className="font-mono text-[10px] text-on-surface-variant">
            {active.n_samples} of {futures.n_samples} independent MC-Dropout rollouts landed here.
          </div>
          <div className="border-t border-outline-variant pt-2 font-mono text-[9px] leading-relaxed text-on-surface-variant opacity-80">
            Per-branch driving features aren't available -- gradient explainability is computed once, for the single deterministic
            prediction, not per stochastic sample (see explain/explainer.py). The explainability panel on Mode 1 reflects the current
            forecast overall, not this specific branch.
          </div>
          <div className="border-t border-outline-variant pt-2 font-mono text-[9px] leading-relaxed text-on-surface-variant">
            To test whether a defensive action changes this forecast, use the Counterfactual Simulator above -- there's no automatic
            per-branch action mapping (yet), so pick an action there directly rather than trusting a suggested one here.
          </div>
        </div>
      )}
    </Panel>
  )
}
