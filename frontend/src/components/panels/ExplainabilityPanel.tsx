import { useState } from "react"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { IconBarChart, IconExplainability, IconList } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import type { Explanation } from "../../lib/types"

type ViewMode = "chart" | "list"

function FeatureRows({ items }: { items: { feature_name: string; score: number }[] }) {
  const max = Math.max(...items.map((i) => i.score), 1e-9)
  return (
    <ol>
      {items.map((item) => (
        <li key={item.feature_name} className="flex items-center gap-2 border-b border-outline-variant/60 px-3 py-1.5 font-mono text-[10px] last:border-b-0">
          <span className="flex-1 truncate text-on-surface">{item.feature_name}</span>
          <div className="h-1 w-14 shrink-0 bg-surface-container">
            <div className="h-1 bg-primary-container" style={{ width: `${Math.max(4, (item.score / max) * 100)}%` }} />
          </div>
          <DataReadout className="w-14 shrink-0 text-right">{item.score.toFixed(4)}</DataReadout>
        </li>
      ))}
    </ol>
  )
}

/**
 * Horizontal bar chart -- the at-a-glance primary view (per the task's own
 * framing) of the same attribution scores FeatureRows already lists in
 * ranked-list form. Row height scales with item count so 10 bars stay
 * readable without the chart competing for space with its sibling column.
 */
function FeatureBarChart({ items }: { items: { feature_name: string; score: number }[] }) {
  // Descending -- Recharts' vertical BarChart renders the first data item
  // at the top row, and the highest score needs to be that top row to
  // match FeatureRows' own top-ranked-first ordering (verified live: an
  // ascending sort here put the LOWEST score at the top instead).
  const rows = [...items].sort((a, b) => b.score - a.score)
  return (
    <div style={{ height: Math.max(120, rows.length * 22) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#d4d4d8" strokeDasharray="1 6" horizontal={false} />
          <XAxis type="number" tick={{ fill: "#71717a", fontFamily: "JetBrains Mono", fontSize: 9 }} axisLine={{ stroke: "#d4d4d8" }} tickLine={{ stroke: "#d4d4d8" }} />
          <YAxis
            type="category"
            dataKey="feature_name"
            width={92}
            tick={{ fill: "#09090b", fontFamily: "JetBrains Mono", fontSize: 9 }}
            axisLine={{ stroke: "#d4d4d8" }}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "#f4f4f5" }}
            contentStyle={{ background: "#ffffff", border: "1px solid #d4d4d8", fontFamily: "JetBrains Mono", fontSize: 11 }}
            labelStyle={{ color: "#71717a" }}
            formatter={(value) => [typeof value === "number" ? value.toFixed(4) : String(value), "attribution"]}
          />
          <Bar dataKey="score" fill="#09090b" barSize={12} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/**
 * Real GNN attention + gradient-based feature attribution (Phase 5) --
 * the mockup's fabricated feature names ("Kerberos Ticket Requests/min",
 * "RPC Endpoint Mapper Anomaly") are replaced with the model's actual
 * top node/edge features and top attention-weighted flows.
 *
 * Chart view (horizontal bars) is the default, at-a-glance read; the
 * ranked list remains one toggle away as supporting detail for someone who
 * wants the exact ordering and inline mini-bars at a glance too -- neither
 * replaces the other, per the visualization brief.
 */
export function ExplainabilityPanel({ explanation }: { explanation: Explanation }) {
  const [view, setView] = useState<ViewMode>("chart")

  return (
    <Panel
      title="Explainability // Attention Weights"
      icon={<IconExplainability size={13} className="text-primary-container" />}
      aside={
        <div className="flex items-center gap-3">
          <div className="flex border border-outline-variant">
            <button
              onClick={() => setView("chart")}
              title="Bar chart view"
              className={`flex items-center px-1.5 py-1 transition-colors ${view === "chart" ? "bg-surface-container text-on-surface" : "text-on-surface-variant hover:text-on-surface"}`}
            >
              <IconBarChart size={12} />
            </button>
            <button
              onClick={() => setView("list")}
              title="Ranked list view"
              className={`flex items-center border-l border-outline-variant px-1.5 py-1 transition-colors ${view === "list" ? "bg-surface-container text-on-surface" : "text-on-surface-variant hover:text-on-surface"}`}
            >
              <IconList size={12} />
            </button>
          </div>
          <span className="font-mono text-[10px] text-secondary-fixed-dim">rollout: valid</span>
        </div>
      }
      className="reticle-corner flex flex-1 flex-col overflow-hidden"
      bodyClassName="flex flex-1 flex-col overflow-y-auto"
    >
      <p className="border-b border-outline-variant/60 px-3 py-2 font-mono text-[11px] leading-relaxed text-on-surface-variant">
        {explanation.summary}
      </p>
      <div className="grid grid-cols-2 divide-x divide-[color:var(--color-outline-variant)]">
        <div>
          <div className="px-3 py-1.5 font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">host features</div>
          {view === "chart" ? <FeatureBarChart items={explanation.top_node_features} /> : <FeatureRows items={explanation.top_node_features} />}
        </div>
        <div>
          <div className="px-3 py-1.5 font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">flow features</div>
          {view === "chart" ? <FeatureBarChart items={explanation.top_edge_features} /> : <FeatureRows items={explanation.top_edge_features} />}
        </div>
      </div>

      {explanation.top_edges_by_attention.length > 0 && (
        <div className="mt-1 border-t border-outline-variant/60 px-3 py-2">
          <div className="mb-1.5 font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">
            top attention-weighted flows
          </div>
          <table className="w-full border-collapse text-left font-mono text-[10px]">
            <thead>
              <tr className="border-b border-outline-variant text-on-surface-variant">
                <th className="px-1.5 py-1 font-medium">flow</th>
                <th className="px-1.5 py-1 text-right font-medium">weight</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/40">
              {explanation.top_edges_by_attention.slice(0, 5).map((f, i) => (
                <tr key={i} className={f.is_attack_edge ? "text-error" : "text-on-surface"}>
                  <td className="px-1.5 py-1 font-semibold">
                    {f.src_ip} → {f.dst_ip}
                  </td>
                  <td className="px-1.5 py-1 text-right font-bold">{f.attention_weight.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}
