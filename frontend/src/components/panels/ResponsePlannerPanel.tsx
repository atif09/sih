import { useState } from "react"
import { IconAlert, IconArrowDown, IconArrowUp, IconRun, IconTune } from "../../lib/icons"
import { planResponse } from "../../lib/api"
import type { PlannedActionItem } from "../../lib/types"
import { Badge } from "../primitives/Badge"
import { DataReadout } from "../primitives/DataReadout"
import { Panel } from "../primitives/Panel"

type RunStatus = "idle" | "running" | "error"

/**
 * Response Planner (differentiator): automates the Counterfactual
 * Intervention Simulator (see CounterfactualPanel) across every candidate
 * action instead of one at a time -- see counterfactual/planner.py. Reuses
 * the same cached server-side context, no separate upload.
 *
 * Run-on-click, not run-on-mount: each candidate action costs one full
 * K-step rollout (same cost as one CounterfactualPanel run), so a capture
 * with several flagged flows means several rollouts back to back -- worth
 * asking for before spending that compute automatically.
 */
export function ResponsePlannerPanel({ defaultK }: { defaultK: number }) {
  const [status, setStatus] = useState<RunStatus>("idle")
  const [error, setError] = useState<string | null>(null)
  const [actions, setActions] = useState<PlannedActionItem[] | null>(null)

  async function run() {
    setStatus("running")
    setError(null)
    try {
      const res = await planResponse({ k: defaultK })
      setActions(res.actions)
      setStatus("idle")
    } catch (err) {
      setStatus("error")
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail || (err as Error).message || "planning failed")
    }
  }

  return (
    <Panel
      title="Response Planner"
      icon={<IconTune size={13} className="text-primary-container" />}
      aside={<span className="font-mono text-[10px] text-primary-container">[ranks every candidate action]</span>}
      className="reticle-corner flex flex-1 flex-col overflow-hidden"
      bodyClassName="flex flex-1 flex-col gap-3 overflow-y-auto p-4"
    >
      <button
        onClick={() => void run()}
        disabled={status === "running"}
        className="flex items-center justify-center gap-2 bg-primary-container px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-surface-container-lowest transition-opacity hover:opacity-90 disabled:opacity-40"
      >
        <IconRun size={13} />
        {status === "running" ? "running every candidate action..." : actions ? "re-run planner" : "run planner"}
      </button>

      {status === "error" && (
        <p className="flex items-center gap-1.5 font-mono text-[11px] text-error">
          <IconAlert size={12} /> {error}
        </p>
      )}

      {!actions && status !== "error" && (
        <p className="pt-2 text-center font-mono text-[11px] text-on-surface-variant">
          runs isolate/block against every flow the model's attention flagged, ranked by measured risk reduction
        </p>
      )}

      {actions && actions.length === 0 && (
        <p className="pt-2 text-center font-mono text-[11px] text-on-surface-variant">no flagged flows in this context to build candidate actions from</p>
      )}

      {actions && actions.length > 0 && (
        <>
          <table className="w-full border-collapse text-left font-mono text-[10px]">
            <thead>
              <tr className="border-b border-outline-variant text-on-surface-variant">
                <th className="px-1.5 py-1 font-medium">action</th>
                <th className="px-1.5 py-1 text-right font-medium">risk reduction</th>
                <th className="px-1.5 py-1 text-right font-medium">normal flows cut</th>
                <th className="px-1.5 py-1 font-medium">stage after</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/40">
              {actions.map((a, i) => (
                <tr key={a.action_label} className={i === 0 ? "bg-primary-container/10" : ""}>
                  <td className="px-1.5 py-1.5 font-semibold text-on-surface">{a.action_label}</td>
                  <td className="px-1.5 py-1.5 text-right">
                    <DataReadout className={`inline-flex items-center gap-1 ${a.risk_reduction > 0 ? "text-secondary-fixed-dim" : a.risk_reduction < 0 ? "text-error" : ""}`}>
                      {a.risk_reduction > 0 && <IconArrowDown size={11} />}
                      {a.risk_reduction < 0 && <IconArrowUp size={11} />}
                      {Math.abs(a.risk_reduction).toFixed(4)}
                    </DataReadout>
                  </td>
                  <td className="px-1.5 py-1.5 text-right text-on-surface-variant">{a.normal_flows_disrupted}</td>
                  <td className="px-1.5 py-1.5">
                    <Badge signal={a.modified_stage === a.original_stage ? "warning" : "info"}>{a.modified_stage}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  )
}
