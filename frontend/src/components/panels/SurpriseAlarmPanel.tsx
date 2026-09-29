import { useMemo } from "react"
import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts"
import { IconSurprise } from "../../lib/icons"
import { Badge } from "../primitives/Badge"
import { Panel } from "../primitives/Panel"
import type { SurpriseAlarmSummary, TimelinePoint } from "../../lib/types"
import { ALARM_PERCENTILE } from "../../lib/constants"

/**
 * Surprise Alarm (differentiator): reuses the world model's own existing
 * dynamics-prediction loss (see surprise/score.py) as a novelty signal --
 * no new head, no retraining. Rendered as its own chart, not a second axis
 * on TrajectoryTimeline: percentile (0-100) and infiltration probability
 * (0-1) are different scales measuring different things, and a dual-axis
 * chart would misrepresent that as one comparable quantity.
 *
 * Honest by construction: entries with surprise_percentile === null (an
 * empty-window transition -- see surprise/score.py's own bug note) are
 * skipped from the line rather than plotted as 0, which would fabricate a
 * "not surprising" reading for a window that was never actually scored.
 */
export function SurpriseAlarmPanel({ timeline, summary }: { timeline: TimelinePoint[]; summary: SurpriseAlarmSummary }) {
  const rows = useMemo(
    () =>
      timeline
        .filter((t) => t.surprise_percentile !== null)
        .map((t) => ({
          index: t.window_id,
          percentile: t.surprise_percentile as number,
          alarm: t.surprise_alarm ? (t.surprise_percentile as number) : null,
        })),
    [timeline],
  )

  if (rows.length === 0) {
    return (
      <Panel title="Surprise Alarm" icon={<IconSurprise size={13} className="text-primary-container" />} className="reticle-corner flex flex-1 items-center justify-center">
        <p className="p-4 text-center font-mono text-[11px] leading-relaxed text-on-surface-variant">{summary.explanation}</p>
      </Panel>
    )
  }

  return (
    <Panel
      title="Surprise Alarm"
      icon={<IconSurprise size={13} className="text-primary-container" />}
      aside={
        <Badge signal={summary.alarm_raised ? "critical" : "info"}>
          {summary.alarm_raised ? "novel behavior" : "as expected"}
        </Badge>
      }
      className="reticle-corner flex flex-col"
      bodyClassName="flex flex-col gap-2 p-2"
    >
      <div className="h-28 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" vertical={false} />
            <XAxis dataKey="index" tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }} axisLine={{ stroke: "#d4d4d8" }} tickLine={{ stroke: "#d4d4d8" }} />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 50, ALARM_PERCENTILE, 100]}
              tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }}
              axisLine={{ stroke: "#d4d4d8" }}
              tickLine={{ stroke: "#d4d4d8" }}
              width={28}
            />
            <Tooltip
              contentStyle={{ background: "#ffffff", border: "1px solid #d4d4d8", fontFamily: "JetBrains Mono", fontSize: 11 }}
              labelStyle={{ color: "#71717a" }}
              formatter={(value) => [typeof value === "number" ? `${value.toFixed(1)}th pct` : String(value), "surprise"]}
              labelFormatter={(label) => `window ${label}`}
            />
            <ReferenceLine
              y={ALARM_PERCENTILE}
              stroke="#ffb4ab"
              strokeDasharray="3 3"
              label={{ value: "alarm threshold", position: "insideTopRight", fill: "#ffb4ab", fontFamily: "JetBrains Mono", fontSize: 8 }}
            />
            <Line type="monotone" dataKey="percentile" stroke="#71717a" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Scatter dataKey="alarm" fill="#ffb4ab" shape="circle" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="border-t border-outline-variant/60 pt-2 font-mono text-[10px] leading-relaxed text-on-surface-variant">{summary.explanation}</p>
    </Panel>
  )
}
