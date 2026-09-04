import { TrendingDown, TrendingUp, Minus } from "lucide-react"
import { IconFingerprint } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { Badge } from "../primitives/Badge"
import type { AttackFingerprint } from "../../lib/types"
import type { Signal } from "../../lib/severity"

const SIMILARITY_SIGNAL: Record<string, Signal> = { High: "critical", Medium: "warning", Low: "info" }

function TrendIcon({ trend }: { trend: string }) {
  if (trend === "rising") return <TrendingUp size={12} className="text-error" />
  if (trend === "falling") return <TrendingDown size={12} className="text-secondary-fixed-dim" />
  return <Minus size={12} className="text-on-surface-variant" />
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
      <div className="mt-auto border-t border-outline-variant/60 px-3 py-2 font-mono text-[10px] text-on-surface-variant opacity-70">
        Driven by this model's real attack-type head, grouped by stage (max, not summed -- see fingerprint/attack_fingerprint.py). "Exfiltration" always shows no driver: no CIC-IDS2017 attack type maps to it.
      </div>
    </Panel>
  )
}
