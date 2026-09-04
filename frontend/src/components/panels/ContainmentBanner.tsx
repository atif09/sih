import { IconAlert, IconSafeAction, IconWarningAction } from "../../lib/icons"
import { DataReadout } from "../primitives/DataReadout"
import type { ContainmentEstimate } from "../../lib/types"

function RangeBar({ likely, full }: { likely: [number, number]; full: [number, number] }) {
  const max = Math.max(full[1], likely[1], 1)
  const pct = (v: number) => (v / max) * 100
  const fullLeft = pct(full[0])
  const fullWidth = Math.max(pct(full[1]) - fullLeft, 0.5)
  const likelyLeft = pct(likely[0])
  const likelyWidth = Math.max(pct(likely[1]) - likelyLeft, 1.5)

  return (
    <div className="relative h-2.5 w-full bg-surface-container">
      <div className="absolute top-0 h-2.5 border-y border-outline" style={{ left: `${fullLeft}%`, width: `${fullWidth}%` }} />
      <div className="absolute top-0 h-2.5 bg-warning" style={{ left: `${likelyLeft}%`, width: `${likelyWidth}%` }} />
    </div>
  )
}

/**
 * Phase 9, Feature 2. The mockup's top alert strip used a fabricated
 * countdown ("T-3m 42s") and a fake specific action ("Revoke Kerberos TGT
 * Session #8819") -- replaced with the real estimated range (a RANGE, not
 * a point -- see containment/estimator.py's docstring on why that matters)
 * and the backend's own rules-based recommended action.
 */
export function ContainmentBanner({ containment }: { containment: ContainmentEstimate }) {
  const isActiveStage = !containment.recommended_action.startsWith("No action needed")
  const hasRange = containment.likely_range_minutes !== null && containment.full_observed_range_minutes !== null

  if (!isActiveStage) {
    return (
      <div className="flex items-center gap-3 border border-outline-variant bg-surface-container-low p-3">
        <IconSafeAction size={18} className="shrink-0 text-secondary-fixed-dim" />
        <div className="font-mono text-[11px] text-on-surface-variant">
          <span className="font-semibold text-secondary-fixed-dim">No active mitigation window.</span> Current predicted state:{" "}
          <span className="text-on-surface">{containment.current_stage}</span>.
        </div>
      </div>
    )
  }

  return (
    <div className="reticle-corner border-2 border-error bg-error-container/10 p-3.5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-3">
          <IconAlert size={20} className="mt-0.5 shrink-0 text-error" />
          <div>
            <div className="font-mono text-[11px] font-bold uppercase tracking-wider text-error">
              Active mitigation window // {containment.current_stage}
            </div>
            <div className="mt-1 font-mono text-[11px] text-on-surface">
              <span className="font-semibold">Recommended action: </span>
              {containment.recommended_action}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 border-t border-error/30 pt-2.5">
        {hasRange ? (
          <>
            <RangeBar likely={containment.likely_range_minutes!} full={containment.full_observed_range_minutes!} />
            <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px]">
              <span className="text-on-surface-variant">
                typical{" "}
                <DataReadout>
                  {containment.likely_range_minutes![0].toFixed(1)}-{containment.likely_range_minutes![1].toFixed(1)}
                </DataReadout>{" "}
                min
              </span>
              <span className="text-on-surface-variant">
                observed{" "}
                <DataReadout>
                  {containment.full_observed_range_minutes![0].toFixed(1)}-{containment.full_observed_range_minutes![1].toFixed(1)}
                </DataReadout>{" "}
                min
              </span>
              {!containment.has_sufficient_data && (
                <span className="flex items-center gap-1 text-warning">
                  <IconWarningAction size={11} />
                  low confidence ({containment.n_historical_episodes} episode{containment.n_historical_episodes === 1 ? "" : "s"})
                </span>
              )}
            </div>
          </>
        ) : (
          <p className="font-mono text-[10px] text-on-surface-variant">{containment.rationale}</p>
        )}
      </div>
    </div>
  )
}
