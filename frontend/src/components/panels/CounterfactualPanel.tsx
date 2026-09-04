import { useMemo, useState } from "react"
import { ArrowRight } from "lucide-react"
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { IconIsolate, IconBlockFlow, IconRun, IconTune, IconAlert, IconInfo } from "../../lib/icons"
import { simulate } from "../../lib/api"
import type { FlaggedFlow, InterventionActionKind, SimulateResponse } from "../../lib/types"
import { Badge } from "../primitives/Badge"
import { DataReadout } from "../primitives/DataReadout"
import { Panel } from "../primitives/Panel"
import { stageSignal } from "../../lib/severity"

type RunStatus = "idle" | "running" | "error"

const SELECT_CLASS =
  "flex-1 min-w-0 border border-outline-variant bg-surface-container-lowest px-2 py-1.5 font-mono text-[11px] text-primary focus:border-primary-container focus:outline-none"

function uniqueHosts(flows: FlaggedFlow[]): string[] {
  const ids = new Set<string>()
  for (const f of flows) {
    ids.add(f.src_ip)
    ids.add(f.dst_ip)
  }
  return Array.from(ids).sort()
}

/**
 * Phase 9, Feature 3 -- "World Model Counterfactual Intervention Simulator"
 * per the mockup's framing. Acts on the context the backend cached from the
 * most recent /analyze call. Host choices come from `flagged_flows` (the
 * flows the model's attention actually surfaced), labeled as such rather
 * than implied to be the complete graph.
 */
export function CounterfactualPanel({ flaggedFlows, defaultK }: { flaggedFlows: FlaggedFlow[]; defaultK: number }) {
  const hosts = useMemo(() => uniqueHosts(flaggedFlows), [flaggedFlows])
  const [actionKind, setActionKind] = useState<InterventionActionKind>("isolate_host")
  const [hostIp, setHostIp] = useState(hosts[0] ?? "")
  const [srcIp, setSrcIp] = useState(hosts[0] ?? "")
  const [dstIp, setDstIp] = useState(hosts[1] ?? hosts[0] ?? "")
  const [bidirectional, setBidirectional] = useState(true)
  const [k, setK] = useState(defaultK)
  const [status, setStatus] = useState<RunStatus>("idle")
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SimulateResponse | null>(null)

  const canRun = hosts.length > 0 && (actionKind === "isolate_host" ? !!hostIp : !!srcIp && !!dstIp && srcIp !== dstIp)

  async function runSimulation() {
    setStatus("running")
    setError(null)
    try {
      const res = await simulate(
        actionKind === "isolate_host"
          ? { action: "isolate_host", host_ip: hostIp, k }
          : { action: "block_flow", src_ip: srcIp, dst_ip: dstIp, bidirectional, k },
      )
      setResult(res)
      setStatus("idle")
    } catch (err) {
      setStatus("error")
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail || (err as Error).message || "simulation failed")
    }
  }

  const chartRows = result?.original_probability.map((p, i) => ({ step: i + 1, original: p, modified: result.modified_probability[i] }))
  const meanOriginal = result ? result.original_probability.reduce((a, b) => a + b, 0) / result.original_probability.length : 0
  const meanModified = result ? result.modified_probability.reduce((a, b) => a + b, 0) / result.modified_probability.length : 0

  // Real effect sizes here are usually a few hundredths of a probability
  // point -- a fixed [0, 1] axis flattens that into two indistinguishable
  // lines, so the axis zooms to the actual data range instead.
  const yDomain: [number, number] = (() => {
    if (!result) return [0, 1]
    const all = [...result.original_probability, ...result.modified_probability]
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    const pad = Math.max((hi - lo) * 0.4, 0.03)
    return [Math.max(0, lo - pad), Math.min(1, hi + pad)]
  })()

  // "modified" is drawn dashed on top of "original" specifically so that
  // when an action is a no-op (or near enough), the amber line still shows
  // through the gaps instead of being fully hidden. That's the right fix
  // for the line disappearing, but a dashed line sitting almost exactly on
  // a solid one still looks like a rendering glitch unless it's named --
  // so call it out explicitly whenever the two curves are this close.
  const isNearNoop = !!chartRows?.length && Math.max(...chartRows.map((r) => Math.abs(r.modified - r.original))) < 0.01

  return (
    <Panel
      title="World Model Counterfactual Intervention Simulator"
      icon={<IconTune size={13} className="text-primary-container" />}
      aside={<span className="font-mono text-[10px] text-primary-container">[interactive testbed]</span>}
      className="reticle-corner flex flex-1 flex-col overflow-hidden"
      bodyClassName="flex flex-1 flex-col gap-4 overflow-y-auto p-4"
    >
      {hosts.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-center font-mono text-[12px] text-on-surface-variant">
          no observed flows to act on -- run an analysis on Mode 1: Live Feed first
        </div>
      ) : (
        <>
          <div className="flex border border-outline-variant">
            <button
              onClick={() => setActionKind("isolate_host")}
              className={`flex flex-1 items-center justify-center gap-1.5 border-r border-outline-variant px-3 py-2 font-mono text-[11px] uppercase tracking-wide transition-colors ${
                actionKind === "isolate_host" ? "bg-surface-container text-on-surface" : "text-on-surface-variant hover:text-on-surface"
              }`}
            >
              <IconIsolate size={13} /> Isolate host
            </button>
            <button
              onClick={() => setActionKind("block_flow")}
              className={`flex flex-1 items-center justify-center gap-1.5 px-3 py-2 font-mono text-[11px] uppercase tracking-wide transition-colors ${
                actionKind === "block_flow" ? "bg-surface-container text-on-surface" : "text-on-surface-variant hover:text-on-surface"
              }`}
            >
              <IconBlockFlow size={13} /> Block flow
            </button>
          </div>

          {actionKind === "isolate_host" ? (
            <label className="flex flex-col gap-1">
              <span className="font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">host to isolate (from observed flows)</span>
              <select className={SELECT_CLASS} value={hostIp} onChange={(e) => setHostIp(e.target.value)}>
                {hosts.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <label className="flex flex-1 flex-col gap-1">
                  <span className="font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">from</span>
                  <select className={SELECT_CLASS} value={srcIp} onChange={(e) => setSrcIp(e.target.value)}>
                    {hosts.map((h) => (
                      <option key={h} value={h}>
                        {h}
                      </option>
                    ))}
                  </select>
                </label>
                <ArrowRight size={14} className="mt-4 shrink-0 text-on-surface-variant" />
                <label className="flex flex-1 flex-col gap-1">
                  <span className="font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">to</span>
                  <select className={SELECT_CLASS} value={dstIp} onChange={(e) => setDstIp(e.target.value)}>
                    {hosts.map((h) => (
                      <option key={h} value={h}>
                        {h}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <label className="flex items-center gap-2 font-mono text-[10px] text-on-surface-variant">
                <input type="checkbox" checked={bidirectional} onChange={(e) => setBidirectional(e.target.checked)} className="accent-primary-container" />
                block both directions
              </label>
            </div>
          )}

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 font-mono text-[10px] text-on-surface-variant">
              rollout steps (k)
              <input
                type="number"
                min={1}
                max={20}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-14 border border-outline-variant bg-surface-container-lowest px-2 py-1 text-center font-mono text-[11px] text-primary focus:border-primary-container focus:outline-none"
              />
            </label>
            <button
              onClick={() => void runSimulation()}
              disabled={!canRun || status === "running"}
              className="flex flex-1 items-center justify-center gap-2 bg-primary-container px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-surface-container-lowest transition-opacity hover:opacity-90 disabled:opacity-40"
            >
              <IconRun size={13} />
              {status === "running" ? "running rollout..." : "run simulation"}
            </button>
          </div>

          {status === "error" && (
            <p className="flex items-center gap-1.5 font-mono text-[11px] text-error">
              <IconAlert size={12} /> {error}
            </p>
          )}

          {!result && status !== "error" && (
            <p className="pt-2 text-center font-mono text-[11px] text-on-surface-variant">choose an action and run the simulation to compare outcomes</p>
          )}

          {result && chartRows && (
            <div className="flex flex-1 flex-col gap-3 border-t border-outline-variant pt-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] text-on-surface-variant">{result.action_label}</span>
                <div className="flex items-center gap-1.5">
                  <Badge signal={stageSignal({ stage: result.original_stage, basis: "attack_type" })}>{result.original_stage}</Badge>
                  <ArrowRight size={12} className="text-on-surface-variant" />
                  <Badge signal={stageSignal({ stage: result.modified_stage, basis: "attack_type" })}>{result.modified_stage}</Badge>
                </div>
              </div>

              <div className="h-32 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartRows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" vertical={false} />
                    <XAxis dataKey="step" tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 10 }} axisLine={{ stroke: "#d4d4d8" }} tickLine={{ stroke: "#d4d4d8" }} />
                    <YAxis
                      domain={yDomain}
                      tickFormatter={(v) => (typeof v === "number" ? v.toFixed(2) : String(v))}
                      tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 10 }}
                      axisLine={{ stroke: "#d4d4d8" }}
                      tickLine={{ stroke: "#d4d4d8" }}
                      width={36}
                    />
                    <Tooltip
                      contentStyle={{ background: "#ffffff", border: "1px solid #d4d4d8", fontFamily: "JetBrains Mono", fontSize: 11 }}
                      labelStyle={{ color: "#71717a" }}
                      labelFormatter={(label) => `step ${label}`}
                      formatter={(value, name) => [typeof value === "number" ? value.toFixed(3) : String(value), name === "original" ? "no action" : "with action"]}
                    />
                    <Line type="monotone" dataKey="original" stroke="#f59e0b" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    {/* Dashed, not solid -- when an action is a no-op (or
                        near enough), this line's real values sit almost
                        exactly on top of "original"'s. Drawn last, a solid
                        stroke here would fully occlude the amber line
                        underneath, making it look like only one line
                        exists. Dashing it lets the amber line show through
                        the gaps, matching the same solid/dashed
                        observed-vs-predicted convention already used in
                        TrajectoryTimeline and ReplayPanel. */}
                    <Line type="monotone" dataKey="modified" stroke="#09090b" strokeWidth={2} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="flex items-center justify-between font-mono text-[10px]">
                <span className="flex items-center gap-1.5 text-on-surface-variant">
                  <span className="h-1.5 w-3 bg-warning" /> no action
                </span>
                <span className="flex items-center gap-1.5 text-on-surface-variant">
                  <span className="h-1.5 w-3 bg-primary-container" /> with action
                </span>
              </div>

              {isNearNoop && (
                <p className="flex items-start gap-1.5 font-mono text-[10px] leading-relaxed text-on-surface-variant">
                  <IconInfo size={12} className="mt-px shrink-0" />
                  dashed line tracking the solid one = this action had no measurable effect on the forecast
                </p>
              )}

              <div className="flex items-center justify-between border-t border-outline-variant pt-2 font-mono text-[11px]">
                <span className="text-on-surface-variant">mean probability</span>
                <span className="flex items-center gap-1.5">
                  <DataReadout>{meanOriginal.toFixed(3)}</DataReadout>
                  <ArrowRight size={11} className="text-on-surface-variant" />
                  <DataReadout className={meanModified < meanOriginal ? "text-secondary-fixed-dim" : meanModified > meanOriginal ? "text-error" : ""}>
                    {meanModified.toFixed(3)}
                  </DataReadout>
                </span>
              </div>

              {result.side_effects.length > 0 && (
                <ul className="flex flex-col gap-1.5 border-l-2 border-warning pl-2.5">
                  {result.side_effects.map((effect) => (
                    <li key={effect} className="font-mono text-[10px] leading-relaxed text-on-surface-variant">
                      {effect}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </>
      )}
    </Panel>
  )
}
