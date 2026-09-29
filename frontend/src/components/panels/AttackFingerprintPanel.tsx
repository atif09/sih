import { TrendingDown, TrendingUp, Minus } from "lucide-react"
import { Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts"
import { IconFingerprint } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { Badge } from "../primitives/Badge"
import { MITRE_SIX_STAGES, type AttackFingerprint } from "../../lib/types"
import type { Signal } from "../../lib/severity"

const SIMILARITY_SIGNAL: Record<string, Signal> = { High: "critical", Medium: "warning", Low: "info" }

function TrendIcon({ trend }: { trend: string }) {
  if (trend === "rising") return <TrendingUp size={12} className="text-error" />
  if (trend === "falling") return <TrendingDown size={12} className="text-secondary-fixed-dim" />
  return <Minus size={12} className="text-on-surface-variant" />
}

/**
 * Radar/spider view of the same current-vs-predicted comparison the table
 * below lists row by row -- an at-a-glance shape instead of reading six
 * numbers one at a time. Axes are the model's real 6 MITRE stages, always
 * in kill-chain order (not `comparison`'s own delta-sorted order) so the
 * shape is geometrically comparable capture to capture.
 *
 * Current = solid cyan, predicted-future = dashed, matching the SAME
 * observed-vs-predicted convention as every other chart in this console
 * (TrajectoryTimeline, ReplayPanel, CounterfactualPanel) -- not the `#58a6ff`
 * blue this feature's original spec named, which isn't a real color in this
 * build; the app's actual, current convention wins (see CLAUDE.md's
 * Frontend Design System section).
 */
function FingerprintRadar({ fingerprint }: { fingerprint: AttackFingerprint }) {
  const byStage = new Map(fingerprint.comparison.map((c) => [c.stage, c]))
  const rows = MITRE_SIX_STAGES.map((stage) => {
    const c = byStage.get(stage)
    return { stage, current: c?.current_score ?? 0, predicted: c?.predicted_score ?? 0 }
  })

  // Real scores are frequently all "LOW" (well under 0.1) -- see the
  // Honest Limit column this feature ships with. A fixed [0, 1] domain
  // makes that shape collapse to an invisible dot at the center, which
  // reads as "broken," not "low." Zooming the radial axis to the actual
  // data range (same fix already used in CounterfactualPanel's yDomain)
  // keeps the axis linear and honest while making real small differences
  // visible; a floor keeps a flat-zero capture from dividing by ~0.
  const maxValue = Math.max(...rows.flatMap((r) => [r.current, r.predicted]), 0)
  const radiusDomain: [number, number] = [0, Math.max(maxValue * 1.25, 0.05)]

  return (
    <div className="h-56 w-full shrink-0 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={rows} outerRadius="70%">
          <PolarGrid stroke="#d4d4d8" />
          <PolarAngleAxis dataKey="stage" tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 8.5 }} />
          <PolarRadiusAxis domain={radiusDomain} tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 8 }} tickCount={3} axisLine={false} tickFormatter={(v) => (typeof v === "number" ? v.toFixed(2) : String(v))} />
          <Tooltip
            contentStyle={{ background: "#ffffff", border: "1px solid #d4d4d8", fontFamily: "JetBrains Mono", fontSize: 11 }}
            labelStyle={{ color: "#71717a" }}
            formatter={(value, name) => [typeof value === "number" ? value.toFixed(3) : String(value), name === "current" ? "current" : "predicted future"]}
          />
          <Legend
            wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "#71717a" }}
            formatter={(value) => (value === "current" ? "current" : "predicted future")}
          />
          <Radar name="current" dataKey="current" stroke="#00f0ff" strokeWidth={2} fill="#00f0ff" fillOpacity={0.2} dot={{ r: 3, fill: "#00f0ff", stroke: "#ffffff", strokeWidth: 1 }} isAnimationActive={false} />
          <Radar name="predicted" dataKey="predicted" stroke="#ffb4ab" strokeWidth={1.75} strokeDasharray="5 4" fill="#ffb4ab" fillOpacity={0.12} dot={{ r: 3, fill: "#ffb4ab", stroke: "#ffffff", strokeWidth: 1 }} isAnimationActive={false} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

/**
 * Phase 10 differentiator -- Explainable Attack Fingerprint. Uses this
 * model's real 6 MITRE stages as its dimensions (the brief's illustrative
 * ATT&CK tactic list -- Credential Access, Persistence... -- doesn't match
 * anything this model actually predicts; see fingerprint/attack_fingerprint.py
 * for the full reasoning). Shows current vs. predicted-future similarity
 * per stage, and names which real attack-type signal drove each score.
 */
export function AttackFingerprintPanel({ fingerprint }: { fingerprint: AttackFingerprint }) {
  return (
    <Panel
      title="Explainable Attack Fingerprint"
      icon={<IconFingerprint size={13} className="text-primary-container" />}
      className="reticle-corner flex flex-1 flex-col overflow-hidden"
      bodyClassName="flex flex-1 flex-col overflow-y-auto"
    >
      <p className="border-b border-outline-variant/60 px-3 py-2 font-mono text-[11px] leading-relaxed text-on-surface-variant">
        {fingerprint.summary}
      </p>
      <div className="grid grid-cols-1 divide-y divide-outline-variant/60 lg:grid-cols-12 lg:divide-x lg:divide-y-0">
        <div className="lg:col-span-5">
          <FingerprintRadar fingerprint={fingerprint} />
        </div>
        <div className="lg:col-span-7">
          <table className="w-full border-collapse text-left font-mono text-[11px]">
            <thead>
              <tr className="border-b border-outline-variant text-on-surface-variant">
                <th className="px-3 py-1.5 font-medium">stage</th>
                <th className="px-3 py-1.5 font-medium">current</th>
                <th className="px-3 py-1.5 font-medium">predicted</th>
                <th className="px-3 py-1.5 text-center font-medium">trend</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/40">
              {fingerprint.comparison.map((entry) => (
                <tr key={entry.stage}>
                  <td className="px-3 py-1.5 text-on-surface">{entry.stage}</td>
                  <td className="px-3 py-1.5">
                    <Badge signal={SIMILARITY_SIGNAL[entry.current_similarity]}>{entry.current_similarity}</Badge>
                  </td>
                  <td className="px-3 py-1.5">
                    <Badge signal={SIMILARITY_SIGNAL[entry.predicted_similarity]}>{entry.predicted_similarity}</Badge>
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center justify-center">
                      <TrendIcon trend={entry.trend} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="mt-auto border-t border-outline-variant/60 px-3 py-2 font-mono text-[10px] text-on-surface-variant opacity-70">
        Driven by this model's real attack-type head, grouped by stage (max, not summed -- see fingerprint/attack_fingerprint.py). "Exfiltration" always shows no driver: no CIC-IDS2017 attack type maps to it.
      </div>
    </Panel>
  )
}
