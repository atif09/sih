import { IconExplainability } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import type { Explanation } from "../../lib/types"

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
 * Real GNN attention + gradient-based feature attribution (Phase 5) --
 * the mockup's fabricated feature names ("Kerberos Ticket Requests/min",
 * "RPC Endpoint Mapper Anomaly") are replaced with the model's actual
 * top node/edge features and top attention-weighted flows.
 */
export function ExplainabilityPanel({ explanation }: { explanation: Explanation }) {
  return (
    <Panel
      title="Explainability // Attention Weights"
      icon={<IconExplainability size={13} className="text-primary-container" />}
      aside={<span className="font-mono text-[10px] text-secondary-fixed-dim">rollout: valid</span>}
      className="reticle-corner flex flex-1 flex-col overflow-hidden"
      bodyClassName="flex flex-1 flex-col overflow-y-auto"
    >
      <p className="border-b border-outline-variant/60 px-3 py-2 font-mono text-[11px] leading-relaxed text-on-surface-variant">
        {explanation.summary}
      </p>
      <div className="grid grid-cols-2 divide-x divide-[color:var(--color-outline-variant)]">
        <div>
          <div className="px-3 py-1.5 font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">host features</div>
          <FeatureRows items={explanation.top_node_features} />
        </div>
        <div>
          <div className="px-3 py-1.5 font-mono text-[9px] uppercase tracking-wide text-on-surface-variant">flow features</div>
          <FeatureRows items={explanation.top_edge_features} />
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
