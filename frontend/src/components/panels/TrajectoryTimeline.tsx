import { useMemo } from "react"
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { IconTrajectory } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import type { ForecastPoint, TimelinePoint } from "../../lib/types"

/**
 * Observed -> predicted probability trajectory. Continuous x-axis of window
 * index, a "now" marker separating real observed history (solid line) from
 * the K-step forecast (dashed, cold-cyan -- the mockup's "predicted state"
 * accent). The mockup's "95% Bayesian confidence band" around the forecast
 * is dropped -- this model doesn't produce calibrated uncertainty bounds
 * (no variance/quantile head), so shading a fake band would misrepresent
 * a single point-estimate rollout as something it isn't.
 */
export function TrajectoryTimeline({ timeline, forecast }: { timeline: TimelinePoint[]; forecast: ForecastPoint[] }) {
  const { rows, nowIndex, peak } = useMemo(() => {
    const lastWindow = timeline.length ? timeline[timeline.length - 1].window_id : 0
    const observedRows = timeline.map((t) => ({
      index: t.window_id,
      observed: t.infiltration_probability,
      predicted: null as number | null,
    }))
    const bridge = timeline.length
      ? [{ index: lastWindow, observed: null, predicted: timeline[timeline.length - 1].infiltration_probability }]
      : []
    const forecastRows = forecast.map((f) => ({
      index: lastWindow + f.step,
      observed: null as number | null,
      predicted: f.infiltration_probability,
    }))
    const peak = forecast.reduce(
      (best, f) => (f.infiltration_probability > best.value ? { value: f.infiltration_probability, step: f.step } : best),
      { value: -Infinity, step: 0 },
    )
    return { rows: [...observedRows, ...bridge, ...forecastRows], nowIndex: lastWindow, peak }
  }, [timeline, forecast])

  const minIndex = rows[0]?.index ?? 0
  const maxIndex = rows[rows.length - 1]?.index ?? 0
  // nowIndex is the last REAL observed window_id -- 0 when there wasn't
  // enough capture to build any walk-forward history at all (every window
  // got consumed as model context instead), which sits *before* minIndex
  // and would otherwise push the "now" line and the shaded zones off the
  // left edge of the chart, outside its actual [minIndex, maxIndex] domain.
  // Clamping means: no observed history -> the OBSERVED zone correctly
  // collapses to nothing and FORECAST correctly covers the whole chart,
  // rather than both silently failing to render.
  const clampedNow = Math.min(Math.max(nowIndex, minIndex), maxIndex)

  return (
    <Panel
      title="Attack Probability Trajectory"
      icon={<IconTrajectory size={13} className="text-primary-container" />}
      aside={
        peak.value > -Infinity ? (
          <span className="font-mono text-[10px] font-bold text-error">
            peak forecast: {peak.value.toFixed(3)} at step {peak.step}
          </span>
        ) : undefined
      }
      className="reticle-corner flex flex-col"
      bodyClassName="h-44 shrink-0 p-2"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 12, right: 24, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="observedFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00f0ff" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#00f0ff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" vertical={false} />
          {/* Explicit observed/forecast zones -- previously only the
              solid-vs-dashed line style distinguished them, which assumes
              the viewer already knows that convention. A shaded background
              + printed label makes it unmistakable without narration. */}
          <ReferenceArea
            x1={minIndex}
            x2={clampedNow}
            fill="#00f0ff"
            fillOpacity={0.035}
            label={{ value: "OBSERVED", position: "insideTopLeft", fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }}
          />
          <ReferenceArea
            x1={clampedNow}
            x2={maxIndex}
            fill="#ffb4ab"
            fillOpacity={0.08}
            label={{ value: "FORECAST", position: "insideBottomRight", fill: "#ffb4ab", fontFamily: "JetBrains Mono", fontSize: 9, fontWeight: 700 }}
          />
          <XAxis
            dataKey="index"
            type="number"
            domain={["dataMin", "dataMax"]}
            allowDuplicatedCategory={false}
            tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 10 }}
            axisLine={{ stroke: "#d4d4d8" }}
            tickLine={{ stroke: "#d4d4d8" }}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 10 }}
            axisLine={{ stroke: "#d4d4d8" }}
            tickLine={{ stroke: "#d4d4d8" }}
            width={32}
          />
          <Tooltip
            contentStyle={{ background: "#ffffff", border: "1px solid #d4d4d8", fontFamily: "JetBrains Mono", fontSize: 11 }}
            labelStyle={{ color: "#71717a" }}
            formatter={(value, name) => [
              typeof value === "number" ? value.toFixed(3) : String(value),
              name === "observed" ? "observed" : "predicted",
            ]}
            labelFormatter={(label) => `window ${label}`}
          />
          <ReferenceLine y={0.5} stroke="#d4d4d8" strokeDasharray="2 4" />
          <ReferenceLine
            x={clampedNow}
            stroke="#09090b"
            strokeWidth={1}
            label={{ value: "now", position: "insideTopRight", fill: "#09090b", fontFamily: "JetBrains Mono", fontSize: 10 }}
          />
          <Area type="monotone" dataKey="observed" stroke="none" fill="url(#observedFill)" isAnimationActive animationDuration={600} />
          <Line type="monotone" dataKey="observed" stroke="#00f0ff" strokeWidth={2} dot={false} isAnimationActive animationDuration={600} />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#ffb4ab"
            strokeWidth={2}
            strokeDasharray="5 4"
            strokeOpacity={0.9}
            dot={false}
            isAnimationActive
            animationDuration={600}
            animationBegin={400}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Panel>
  )
}
