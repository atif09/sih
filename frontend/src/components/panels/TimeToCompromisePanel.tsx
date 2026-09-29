import { useMemo } from "react"
import { Area, CartesianGrid, ComposedChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { IconClock } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import type { TimeToCompromise } from "../../lib/types"

/**
 * Time-to-Compromise (differentiator): cumulative fraction of the SAME
 * MC-Dropout ensemble Multiple Parallel Futures already samples that has
 * crossed the attack threshold by minute k -- see
 * futures/time_to_compromise.py. Built from real Monte-Carlo sampling, not
 * a smooth analytic curve -- the step shape IS the real data (n=40
 * samples), not a rendering artifact.
 */
export function TimeToCompromisePanel({ ttc }: { ttc: TimeToCompromise }) {
  const rows = useMemo(
    () => ttc.cumulative_fraction.map((frac, i) => ({ step: i + 1, pct: frac * 100 })),
    [ttc.cumulative_fraction],
  )

  return (
    <Panel
      title="Time-to-Compromise"
      icon={<IconClock size={13} className="text-primary-container" />}
      aside={
        ttc.median_crossing_step !== null ? (
          <span className="font-mono text-[10px] font-bold text-error">50% risk reached at minute {ttc.median_crossing_step}</span>
        ) : (
          <span className="font-mono text-[10px] text-on-surface-variant">fewer than half of sampled futures cross threshold in this horizon</span>
        )
      }
      className="reticle-corner flex flex-col"
      bodyClassName="flex flex-col gap-2 p-2"
    >
      <div className="h-32 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="ttcFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ffb4ab" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#ffb4ab" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" vertical={false} />
            <XAxis
              dataKey="step"
              tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }}
              axisLine={{ stroke: "#d4d4d8" }}
              tickLine={{ stroke: "#d4d4d8" }}
              label={{ value: "minute", position: "insideBottom", offset: -2, fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }}
              axisLine={{ stroke: "#d4d4d8" }}
              tickLine={{ stroke: "#d4d4d8" }}
              width={30}
            />
            <Tooltip
              contentStyle={{ background: "#ffffff", border: "1px solid #d4d4d8", fontFamily: "JetBrains Mono", fontSize: 11 }}
              labelStyle={{ color: "#71717a" }}
              formatter={(value) => [typeof value === "number" ? `${value.toFixed(1)}%` : String(value), "compromised by"]}
              labelFormatter={(label) => `minute ${label}`}
            />
            <ReferenceLine y={50} stroke="#d4d4d8" strokeDasharray="2 4" />
            <Area type="stepAfter" dataKey="pct" stroke="#ffb4ab" strokeWidth={2} fill="url(#ttcFill)" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between border-t border-outline-variant pt-2 font-mono text-[10px]">
        <span className="text-on-surface-variant">never crossed within horizon</span>
        <DataReadout>{(ttc.never_crossed_fraction * 100).toFixed(1)}%</DataReadout>
      </div>
    </Panel>
  )
}
