import { useEffect, useMemo, useRef, useState } from "react"
import { CartesianGrid, Line, ComposedChart, ReferenceDot, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { IconPause, IconRewind, IconRun } from "../../lib/icons"
import { Badge } from "../primitives/Badge"
import { DataReadout } from "../primitives/DataReadout"
import { Panel } from "../primitives/Panel"
import { TopologyGraph } from "./TopologyGraph"
import { stageSignal } from "../../lib/severity"
import type { AnalyzeResponse, FlaggedFlow } from "../../lib/types"

interface ReplayStep {
  kind: "observed" | "predicted"
  label: string
  probability: number
  stage: string
  basis: string
  flows: FlaggedFlow[]
}

const PLAYBACK_INTERVAL_MS = 900

/**
 * Attack Future Replay (Phase 10 differentiator, priority #1 per the
 * reference brief): scrub through observed windows and the K-step
 * forecast as one continuous timeline. Every OBSERVED step shows its own
 * real per-window attention (Attack Future Replay's backend addition --
 * see api/pipeline.py). PREDICTED steps have no equivalent: the world
 * model predicts a latent embedding for the future, never a decoded
 * graph (see rollout/engine.py's own docstring) -- so the topology stays
 * frozen on the last real observed graph during predicted steps, clearly
 * labeled, rather than fabricating future structure.
 */
export function ReplayPanel({ result }: { result: AnalyzeResponse }) {
  const steps = useMemo<ReplayStep[]>(() => {
    const observed: ReplayStep[] = result.timeline.map((t) => ({
      kind: "observed",
      label: `window ${t.window_id}`,
      probability: t.infiltration_probability,
      stage: t.mitre_stage.stage,
      basis: t.mitre_stage.basis,
      flows: t.top_flows,
    }))
    const lastObservedFlows = observed.length ? observed[observed.length - 1].flows : []
    const predicted: ReplayStep[] = result.forecast.map((f) => ({
      kind: "predicted",
      label: `+${f.step} step${f.step === 1 ? "" : "s"}`,
      probability: f.infiltration_probability,
      stage: f.mitre_stage.stage,
      basis: f.mitre_stage.basis,
      flows: lastObservedFlows,
    }))
    return [...observed, ...predicted]
  }, [result])

  const [index, setIndex] = useState(Math.max(0, steps.length - 1))
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    setIndex(Math.max(0, steps.length - 1))
  }, [steps])

  // The topology graph lags the scrubber by a short debounce -- see the
  // dedicated comment below where it's consumed for why.
  const [graphIndex, setGraphIndex] = useState(Math.max(0, steps.length - 1))
  useEffect(() => {
    setGraphIndex(Math.max(0, steps.length - 1))
  }, [steps])
  useEffect(() => {
    const t = window.setTimeout(() => setGraphIndex(index), 150)
    return () => window.clearTimeout(t)
  }, [index])

  useEffect(() => {
    if (!playing) return
    timerRef.current = window.setInterval(() => {
      setIndex((i) => {
        if (i >= steps.length - 1) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, PLAYBACK_INTERVAL_MS)
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [playing, steps.length])

  const current = steps[index]
  // Rendered graph lags the scrubber (see graphIndex above): rapid slider
  // drags -- especially while Play is also running -- send react-force-graph
  // a brand-new `flows` array every few ms, and it treats each one as
  // entirely new graph data, restarting its d3 physics simulation from
  // scratch each time. If that happens faster than the simulation can
  // settle, `onEngineStop` (which calls zoomToFit) never fires and nodes
  // are left positioned outside the visible viewport -- the graph appears
  // to "go blank". Only the graph reads the debounced index; the
  // probability/stage readouts below still read `current` directly so they
  // stay instantly responsive while scrubbing.
  const graphStep = steps[Math.min(graphIndex, steps.length - 1)] ?? current
  const chartRows = steps.map((s, i) => ({
    index: i,
    observed: s.kind === "observed" ? s.probability : null,
    predicted: s.kind === "predicted" ? s.probability : null,
  }))
  // Bridge point so the predicted dashed line visually connects to the
  // last observed point instead of floating from nothing.
  if (result.timeline.length && result.forecast.length) {
    chartRows[result.timeline.length - 1].predicted = chartRows[result.timeline.length - 1].observed
  }

  if (steps.length === 0) {
    return (
      <Panel title="Attack Future Replay" className="reticle-corner flex flex-1 items-center justify-center">
        <p className="p-6 text-center font-mono text-[12px] text-on-surface-variant">no timeline data to replay</p>
      </Panel>
    )
  }

  return (
    <div className="flex flex-1 flex-col gap-3">
      <Panel className="reticle-corner" bodyClassName="flex flex-col gap-3 p-3">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => {
              setPlaying(false)
              setIndex(0)
            }}
            title="Rewind to start"
            className="flex items-center justify-center border border-outline-variant bg-surface-container p-2 text-on-surface-variant transition-colors hover:text-on-surface"
          >
            <IconRewind size={14} />
          </button>
          <button
            onClick={() => setPlaying((p) => !p)}
            className="flex items-center gap-1.5 bg-primary-container px-3 py-2 font-mono text-[11px] font-bold uppercase tracking-wider text-surface-container-lowest transition-opacity hover:opacity-90"
          >
            {playing ? <IconPause size={13} /> : <IconRun size={13} />}
            {playing ? "pause" : "play"}
          </button>
          <input
            type="range"
            min={0}
            max={steps.length - 1}
            value={index}
            onChange={(e) => {
              setPlaying(false)
              setIndex(Number(e.target.value))
            }}
            className="min-w-40 flex-1 accent-primary-container"
          />
          <div className="flex items-center gap-2 font-mono text-[11px] text-on-surface-variant">
            <Badge signal={current.kind === "observed" ? "info" : "warning"}>{current.kind}</Badge>
            <span>{current.label}</span>
          </div>
        </div>

        <div className="h-24 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartRows} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" vertical={false} />
              <XAxis dataKey="index" tick={false} axisLine={{ stroke: "#d4d4d8" }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }} axisLine={{ stroke: "#d4d4d8" }} tickLine={{ stroke: "#d4d4d8" }} width={28} />
              <ReferenceLine x={result.timeline.length - 1} stroke="#09090b" strokeDasharray="2 3" />
              <Line type="monotone" dataKey="observed" stroke="#00f0ff" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="predicted" stroke="#ffb4ab" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
              <ReferenceDot x={index} y={current.probability} r={4} fill={current.kind === "observed" ? "#09090b" : "#ffb4ab"} stroke="#ffffff" strokeWidth={1.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="grid flex-1 grid-cols-1 gap-3 lg:grid-cols-12">
        <div className="lg:col-span-8">
          <TopologyGraph flows={graphStep.flows} />
        </div>
        <Panel title="Replay State" className="reticle-corner lg:col-span-4" bodyClassName="flex flex-col gap-3 p-4">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">infiltration probability</div>
            <DataReadout className="text-3xl font-bold">{current.probability.toFixed(3)}</DataReadout>
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">mitre stage</div>
            <Badge signal={stageSignal({ stage: current.stage, basis: current.basis })}>{current.stage}</Badge>
          </div>
          {current.kind === "predicted" && (
            <p className="border-t border-outline-variant pt-3 font-mono text-[10px] leading-relaxed text-on-surface-variant">
              Topology frozen on the last real observed graph -- the model predicts a latent embedding for future
              windows, never a decoded graph structure, so no new topology exists to show for predicted steps.
            </p>
          )}
        </Panel>
      </div>
    </div>
  )
}
