import { useEffect, useState } from "react"
import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { getCalibration, getROCPRCurves } from "../../lib/api"
import type { CalibrationResponse, ROCPRCurvesResponse } from "../../lib/types"
import { IconAlert, IconTarget } from "../../lib/icons"
import { Panel } from "../primitives/Panel"

const AXIS_TICK = { fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }
const AXIS_LINE = { stroke: "#d4d4d8" }
const TOOLTIP_STYLE = { background: "#ffffff", border: "1px solid #d4d4d8", fontFamily: "JetBrains Mono", fontSize: 11 }
const fmt3 = (v: number) => v.toFixed(3)

function MetricBadge({ label, value, dashed, title }: { label: string; value: number; dashed?: boolean; title?: string }) {
  return (
    <span className="flex items-center gap-1.5 font-mono text-[10px] text-on-surface-variant" title={title}>
      <span className="inline-block h-0.5 w-3" style={{ background: dashed ? "repeating-linear-gradient(90deg, #71717a 0 3px, transparent 3px 5px)" : "#00f0ff" }} />
      {label} <span className="font-bold text-on-surface">{value.toFixed(3)}</span>
    </span>
  )
}

function ROCChart({ data }: { data: ROCPRCurvesResponse }) {
  const wmPoints = data.world_model.roc ? data.world_model.roc.fpr.map((x, i) => ({ x, y: data.world_model.roc!.tpr[i] })) : []
  const blPoints = data.baseline.roc ? data.baseline.roc.fpr.map((x, i) => ({ x, y: data.baseline.roc!.tpr[i] })) : []

  return (
    <Panel
      title="ROC Curve"
      icon={<IconTarget size={13} className="text-primary-container" />}
      aside={
        <div className="flex items-center gap-3">
          <MetricBadge label="world model auc" value={data.world_model.auc} />
          <MetricBadge label="baseline auc" value={data.baseline.auc} dashed />
        </div>
      }
      className="reticle-corner flex flex-col"
      bodyClassName="h-56 shrink-0 p-2"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 12, right: 16, bottom: 8, left: 4 }}>
          <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={fmt3} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={AXIS_LINE} />
          <YAxis type="number" domain={[0, 1]} tickFormatter={fmt3} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={AXIS_LINE} width={34} />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelStyle={{ color: "#71717a" }}
            formatter={(value, name) => [typeof value === "number" ? value.toFixed(3) : String(value), name === "y" ? "true positive rate" : String(name)]}
            labelFormatter={(label) => `false positive rate ${typeof label === "number" ? label.toFixed(3) : label}`}
          />
          {/* Random-classifier reference -- a model with no real signal traces this exact diagonal */}
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#d4d4d8" strokeDasharray="2 4" ifOverflow="extendDomain" />
          <Line data={blPoints} dataKey="y" stroke="#71717a" strokeWidth={1.75} strokeDasharray="5 4" dot={false} isAnimationActive={false} name="baseline" />
          <Line data={wmPoints} dataKey="y" stroke="#00f0ff" strokeWidth={2.25} dot={false} isAnimationActive={false} name="world model" />
        </ComposedChart>
      </ResponsiveContainer>
    </Panel>
  )
}

function PRChart({ data }: { data: ROCPRCurvesResponse }) {
  const wmPoints = data.world_model.pr ? data.world_model.pr.recall.map((x, i) => ({ x, y: data.world_model.pr!.precision[i] })) : []
  const blPoints = data.baseline.pr ? data.baseline.pr.recall.map((x, i) => ({ x, y: data.baseline.pr!.precision[i] })) : []

  return (
    <Panel
      title="Precision-Recall Curve"
      icon={<IconTarget size={13} className="text-primary-container" />}
      aside={
        <div className="flex items-center gap-3">
          <MetricBadge label="world model ap" value={data.world_model.average_precision} title="Average Precision -- area under this precision-recall curve" />
          <MetricBadge label="baseline ap" value={data.baseline.average_precision} dashed title="Average Precision -- area under this precision-recall curve" />
        </div>
      }
      className="reticle-corner flex flex-col"
      bodyClassName="h-56 shrink-0 p-2"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 12, right: 16, bottom: 8, left: 4 }}>
          <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={fmt3} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={AXIS_LINE} />
          <YAxis type="number" domain={[0, 1]} tickFormatter={fmt3} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={AXIS_LINE} width={34} />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelStyle={{ color: "#71717a" }}
            formatter={(value) => [typeof value === "number" ? value.toFixed(3) : String(value), "precision"]}
            labelFormatter={(label) => `recall ${typeof label === "number" ? label.toFixed(3) : label}`}
          />
          <Line data={blPoints} dataKey="y" stroke="#71717a" strokeWidth={1.75} strokeDasharray="5 4" dot={false} isAnimationActive={false} name="baseline" />
          <Line data={wmPoints} dataKey="y" stroke="#00f0ff" strokeWidth={2.25} dot={false} isAnimationActive={false} name="world model" />
        </ComposedChart>
      </ResponsiveContainer>
    </Panel>
  )
}

/**
 * Custom dot that also prints the real sample count behind each point --
 * with only a couple of populated bins (see the component doc below),
 * disclosing "how many real examples is this point resting on" matters
 * more here than it would on a dense curve.
 */
function CalibrationDot(props: { cx?: number; cy?: number; payload?: { n: number } }) {
  const { cx, cy, payload } = props
  if (cx == null || cy == null || !payload) return null
  return (
    <g>
      <circle cx={cx} cy={cy} r={4} fill="#00f0ff" stroke="#09090b" strokeWidth={1} />
      <text x={cx} y={cy - 10} textAnchor="middle" fontFamily="JetBrains Mono" fontSize={9} fill="#71717a">
        n={payload.n}
      </text>
    </g>
  )
}

function CalibrationChart({ data }: { data: CalibrationResponse }) {
  const wm = data.world_model
  const points = wm ? wm.predicted_confidence.map((x, i) => ({ x, y: wm.actual_frequency[i], n: wm.bin_counts[i] })) : []

  return (
    <Panel
      title="Confidence Calibration"
      icon={<IconTarget size={13} className="text-primary-container" />}
      aside={<span className="font-mono text-[10px] text-on-surface-variant">world model only · {data.n_bins} bins configured</span>}
      className="reticle-corner flex flex-col"
      bodyClassName="h-56 shrink-0 p-2"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 16, right: 16, bottom: 8, left: 4 }}>
          <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" />
          {/* No axis-title labels -- matches ROC/PR's own minimal style
              (meaning carried by the panel title + tooltip, not an inline
              axis caption); an earlier attempt at a rotated y-axis label
              here visibly overlapped the tick numbers. */}
          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={fmt3} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={AXIS_LINE} />
          <YAxis type="number" domain={[0, 1]} tickFormatter={fmt3} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={AXIS_LINE} width={34} />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelStyle={{ color: "#71717a" }}
            formatter={(value, name, item) => [
              typeof value === "number" ? value.toFixed(3) : String(value),
              name === "y" ? `actual frequency (n=${(item?.payload as { n: number } | undefined)?.n ?? "?"})` : String(name),
            ]}
            labelFormatter={(label) => `predicted confidence ${typeof label === "number" ? label.toFixed(3) : label}`}
          />
          {/* Perfect-calibration reference -- predicted confidence == actual frequency at every point */}
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#d4d4d8" strokeDasharray="2 4" ifOverflow="extendDomain" />
          {/* Visible dots (unlike every other chart's dot={false}) -- with
              only a couple of real points, a thin connecting line alone
              would misread as a dense, smooth curve rather than 2 measured
              bins. */}
          <Line data={points} dataKey="y" stroke="#00f0ff" strokeWidth={2} dot={<CalibrationDot />} isAnimationActive={false} name="world model" />
        </ComposedChart>
      </ResponsiveContainer>
    </Panel>
  )
}

/**
 * Model Performance -- shown in Mode 1's pre-upload state (this is a
 * MODEL-level property, not tied to any capture, so it doesn't need one).
 * Deliberately placed here rather than as its own nav tab, since a tab for
 * exactly this kind of static benchmark view was built and then removed
 * earlier in this project; showing it as the empty-state content instead
 * keeps it reachable without resurrecting that structural element -- it
 * gives way to the real per-capture panels the moment an analysis exists.
 *
 * ROC alongside PR, always together (never toggled) -- with this data's
 * real class imbalance (attacks are the minority class on the held-out
 * day), ROC alone can look more flattering than the data supports; PR
 * keeps that honest.
 */
export function ModelPerformance() {
  const [data, setData] = useState<ROCPRCurvesResponse | null>(null)
  const [calibrationData, setCalibrationData] = useState<CalibrationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getROCPRCurves()
      .then((res) => {
        if (!cancelled) setData(res)
      })
      .catch((err) => {
        if (!cancelled) {
          const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
          setError(detail || (err as Error).message || "failed to load model performance data")
        }
      })
    getCalibration()
      .then((res) => {
        if (!cancelled) setCalibrationData(res)
      })
      .catch(() => {
        // Non-fatal: ROC/PR still render on their own if only calibration is missing.
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="flex w-full max-w-6xl flex-col gap-4">
      <div className="text-center">
        <h2 className="font-mono text-[11px] font-semibold uppercase tracking-wider text-on-surface">model performance</h2>
        <p className="mt-1 font-mono text-[11px] text-on-surface-variant">
          world model vs. logistic-regression baseline, on {data?.held_out_day ?? "the held-out day"} — an attack type excluded entirely from
          training
        </p>
      </div>

      {error && (
        <p className="flex items-center justify-center gap-1.5 font-mono text-[11px] text-error">
          <IconAlert size={12} /> {error}
        </p>
      )}

      {!error && !data && <p className="text-center font-mono text-[11px] text-on-surface-variant">loading benchmark data...</p>}

      {data && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
          <ROCChart data={data} />
          <PRChart data={data} />
          {calibrationData && <CalibrationChart data={calibrationData} />}
        </div>
      )}

      {calibrationData?.world_model && (
        <p className="text-center font-mono text-[10px] leading-relaxed text-on-surface-variant">
          Finding: the model's raw output values cluster tightly (only {calibrationData.world_model.predicted_confidence.length} of{" "}
          {calibrationData.n_bins} confidence bins have any held-out examples in them at all) and are poorly calibrated as literal
          probabilities — one populated bin reads far below the diagonal, the other far above it, in opposite directions. Read the model's raw
          output as a ranking/classification signal (its thresholded accuracy above is genuinely strong), not as a calibrated confidence
          percentage.
        </p>
      )}
    </div>
  )
}
