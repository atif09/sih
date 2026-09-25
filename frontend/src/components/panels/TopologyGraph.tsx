import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { forceX, forceY } from "d3-force-3d"
import ForceGraph2D, { type ForceGraphMethods, type NodeObject, type LinkObject } from "react-force-graph-2d"
import { IconAlert, IconRadar, IconSafeAction } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import { STAGE_BENIGN } from "../../lib/types"
import type { FlaggedFlow, MitreStage } from "../../lib/types"

interface GraphNode extends NodeObject {
  id: string
  severity: "target" | "suspicious" | "normal"
}
interface GraphLink extends LinkObject {
  source: string
  target: string
  weight: number
  isPath: boolean
}

// Real design-system tokens (see CLAUDE.md's Frontend Design System) -- the
// brief for this feature named #f85149/#d29922, colors that don't exist
// anywhere in this build. Substituted with the real critical/warning
// tokens (SIGNAL_HEX's own values, re-declared as literals here since
// canvas can't consume CSS vars, same pattern already used for every
// other hardcoded chart/canvas color in this file).
const COLOR_NODE = "#00f0ff" // normal -- unchanged, the restored cyan accent
const COLOR_SUSPICIOUS = "#f59e0b" // warning token
const COLOR_TARGET = "#ffb4ab" // critical/error token
const COLOR_LINK = "#4edea3" // secondary-fixed-dim -- unchanged, normal flow edges
const COLOR_SELECTED_RING = "#71717a" // on-surface-variant

interface NodeStats {
  ip: string
  severity: GraphNode["severity"]
  touchingFlows: number
  totalBytes: number
  totalFlowCount: number
  maxAttentionWeight: number
}

function computeNodeStats(ip: string, flows: FlaggedFlow[], severity: GraphNode["severity"]): NodeStats {
  const touching = flows.filter((f) => f.src_ip === ip || f.dst_ip === ip)
  return {
    ip,
    severity,
    touchingFlows: touching.length,
    totalBytes: touching.reduce((a, f) => a + f.bytes_sum, 0),
    totalFlowCount: touching.reduce((a, f) => a + f.flow_count, 0),
    maxAttentionWeight: touching.reduce((a, f) => Math.max(a, f.attention_weight), 0),
  }
}

/**
 * Real GNN attention -- not the `is_attack_edge` ground-truth label
 * (`FlaggedFlow.is_attack_edge`'s own docstring: "always False for a real
 * unlabeled upload"). The old version of this component colored nodes red
 * only via that label, which meant a real analyst upload (the actual
 * deployment case, not a labeled-dataset replay) never highlighted
 * anything at all. Severity here comes from the model's own real output:
 * `attention_concentration`, the same field the ranked flow lists already
 * sort by everywhere else in this app.
 *
 * "Predicted target" = the single highest-attention flow's destination.
 * "Path" = that flow, extended forward through any other top-ranked flows
 * that happen to chain from it (this flow's dst is the next one's src) --
 * real chaining if it exists in the data, never fabricated past what the
 * ranked list actually contains.
 */
function classifySeverity(flows: FlaggedFlow[], isBenign: boolean) {
  const ranked = [...flows].sort((a, b) => b.attention_concentration - a.attention_concentration)
  const targetHost: string | null = !isBenign && ranked.length > 0 ? ranked[0].dst_ip : null

  // Greedily chain forward from the top flow through subsequent top-ranked
  // flows whose source matches the chain's current end -- a real multi-hop
  // path if the data supports one, a single edge if it doesn't. Capped at
  // 4 hops so a coincidental long chain in noisy data can't run away.
  const pathFlows: FlaggedFlow[] = []
  if (!isBenign && ranked.length > 0) {
    pathFlows.push(ranked[0])
    let cursor = ranked[0].dst_ip
    const used = new Set([ranked[0].src_ip + "->" + ranked[0].dst_ip])
    for (let hop = 0; hop < 3; hop++) {
      const next = ranked.find((f) => f.src_ip === cursor && !used.has(f.src_ip + "->" + f.dst_ip))
      if (!next) break
      pathFlows.push(next)
      used.add(next.src_ip + "->" + next.dst_ip)
      cursor = next.dst_ip
    }
  }
  const pathEdgeKeys = new Set(pathFlows.map((f) => f.src_ip + "->" + f.dst_ip))
  const pathHosts = new Set(pathFlows.flatMap((f) => [f.src_ip, f.dst_ip]))

  // "Suspicious" = appears among the next-most-attended real flows (rank
  // 2-6), excluding whatever's already on the path -- a real, if softer,
  // signal, not every host in the graph. Gated on !isBenign: attention
  // weights exist for every window regardless of verdict (attention is
  // "which flows most influenced this prediction," not "which flows are
  // attack-like"), so showing amber dots under a calm "no active threat"
  // banner would contradict it -- only color anything when the model
  // actually predicts an active stage.
  const suspiciousHosts = isBenign
    ? new Set<string>()
    : new Set(ranked.slice(1, 6).flatMap((f) => [f.src_ip, f.dst_ip]).filter((h) => !pathHosts.has(h)))

  return { targetHost, pathEdgeKeys, pathHosts, suspiciousHosts, topFlow: ranked[0] ?? null }
}

/**
 * Full-bleed network topology built from `flagged_flows` -- the flows the
 * model's attention actually weighted, not a claim of complete captured
 * topology (the API doesn't expose that). Clicking a node opens a real
 * inspector HUD (the Stitch mockup's node inspector showed fabricated OS
 * fingerprints and degree stats; this one shows only fields the API
 * actually returns, aggregated across that host's flagged flows).
 *
 * `currentStage` drives the plain-language status banner and the
 * node/edge severity heat-map -- optional so this component still renders
 * standalone (e.g. from ReplayPanel, which passes its own per-step stage).
 */
export function TopologyGraph({ flows, currentStage }: { flows: FlaggedFlow[]; currentStage?: MitreStage }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | undefined>(undefined)
  const [selected, setSelected] = useState<string | null>(null)

  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize({ width, height })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Drives the pulse on the predicted-target node/path only -- one
  // deliberate animated element, not motion scattered across the graph
  // (see CLAUDE.md's Motion guidance). ~12fps is smooth enough for a pulse
  // without forcing the canvas to redraw at full frame rate.
  const [pulseTick, setPulseTick] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => setPulseTick((t) => (t + 1) % 1000), 80)
    return () => window.clearInterval(id)
  }, [])
  const pulse = 0.55 + 0.45 * Math.sin(pulseTick * 0.3) // oscillates ~0.1..1

  const isBenign = !currentStage || currentStage.stage === STAGE_BENIGN

  const { nodes, links, targetHost, suspiciousHosts, topFlow } = useMemo(() => {
    const { targetHost, pathEdgeKeys, pathHosts, suspiciousHosts, topFlow } = classifySeverity(flows, isBenign)
    const nodeIds = new Set<string>()
    for (const f of flows) {
      nodeIds.add(f.src_ip)
      nodeIds.add(f.dst_ip)
    }
    const nodes: GraphNode[] = Array.from(nodeIds).map((id) => ({
      id,
      severity: id === targetHost ? "target" : suspiciousHosts.has(id) || pathHosts.has(id) ? "suspicious" : "normal",
    }))
    const links: GraphLink[] = flows.map((f) => ({
      source: f.src_ip,
      target: f.dst_ip,
      weight: f.attention_weight,
      isPath: pathEdgeKeys.has(f.src_ip + "->" + f.dst_ip),
    }))
    return { nodes, links, targetHost, suspiciousHosts, topFlow }
  }, [flows, isBenign])

  useEffect(() => {
    graphRef.current?.d3Force("x", forceX(0).strength(0.05))
    graphRef.current?.d3Force("y", forceY(0).strength(0.05))
  }, [nodes, links])

  const handleEngineStop = useCallback(() => {
    graphRef.current?.zoomToFit(400, 56)
  }, [])

  const selectedStats = selected
    ? computeNodeStats(selected, flows, selected === targetHost ? "target" : suspiciousHosts.has(selected) ? "suspicious" : "normal")
    : null

  return (
    <Panel
      title="Live Attack Topology Graph"
      icon={<IconRadar size={13} className="text-primary-container" />}
      aside={
        <div className="flex items-center gap-3 font-mono text-[10px]">
          <span className="flex items-center gap-1.5 text-on-surface-variant">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLOR_TARGET }} /> predicted target
          </span>
          <span className="flex items-center gap-1.5 text-on-surface-variant">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLOR_SUSPICIOUS }} /> suspicious
          </span>
          <span className="flex items-center gap-1.5 text-on-surface-variant">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLOR_NODE }} /> normal
          </span>
        </div>
      }
      className="reticle-corner flex flex-1 flex-col overflow-hidden"
      bodyClassName="relative flex flex-1 flex-col overflow-hidden"
    >
      {/* Plain-language status banner -- modeled on ContainmentBanner's
          own two-state (active/quiet) treatment for visual consistency,
          not a new pattern invented for this one panel. */}
      {isBenign ? (
        <div className="flex shrink-0 items-center gap-3 border-b border-outline-variant bg-surface-container-low p-3">
          <IconSafeAction size={18} className="shrink-0 text-secondary-fixed-dim" />
          <div className="font-mono text-[12px] text-on-surface-variant">
            <span className="font-semibold text-secondary-fixed-dim">No active threat predicted.</span>{" "}
            {currentStage ? (
              <>
                Current classification: <span className="text-on-surface">{currentStage.stage}</span>.
              </>
            ) : (
              "Traffic classified as benign."
            )}
          </div>
        </div>
      ) : (
        <div className="shrink-0 border-b-2 border-error bg-error-container/10 p-3">
          <div className="flex items-start gap-3">
            <IconAlert size={20} className="mt-0.5 shrink-0 text-error" />
            <div className="font-mono text-[13px] font-bold leading-snug text-error">
              Predicted: {currentStage!.stage}
              {targetHost && (
                <>
                  {" "}
                  targeting <DataReadout className="text-error">{targetHost}</DataReadout>
                </>
              )}
              {" — "}
              {(currentStage!.confidence * 100).toFixed(0)}% confidence
            </div>
          </div>
        </div>
      )}

      <div
        ref={containerRef}
        className="relative min-h-0 flex-1 overflow-hidden bg-surface-container-lowest bg-center bg-size-[16px_16px] bg-[radial-gradient(var(--color-outline-variant)_1px,transparent_1px)]"
      >
        <div className="pointer-events-none absolute bottom-2 left-2 z-10 border border-outline-variant bg-surface-container-lowest/80 px-2 py-1 font-mono text-[10px] text-outline">
          [click node to inspect] // {flows.length} flows observed
        </div>
        {!isBenign && topFlow && (
          <div className="pointer-events-none absolute right-2 top-2 z-10 flex items-center gap-1.5 font-mono text-[10px] text-error">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-error" />
            attack path active
          </div>
        )}

        {selectedStats && (
          <div className="absolute right-2 top-2 z-20 w-56 border border-primary-container bg-surface-container-lowest/95 p-2.5 font-mono text-[10px] shadow-lg backdrop-blur">
            <div className="flex items-center justify-between border-b border-outline-variant pb-1.5">
              <span className="font-bold text-primary">NODE INSPECT</span>
              <button onClick={() => setSelected(null)} className="text-on-surface-variant hover:text-on-surface">
                ×
              </button>
            </div>
            <div className="mt-1.5 space-y-1">
              <div className="flex justify-between">
                <span className="text-on-surface-variant">HOST:</span>
                <DataReadout className="font-bold">{selectedStats.ip}</DataReadout>
              </div>
              <div className="flex justify-between">
                <span className="text-on-surface-variant">STATUS:</span>
                <span
                  className={
                    selectedStats.severity === "target"
                      ? "font-bold text-error"
                      : selectedStats.severity === "suspicious"
                        ? "font-bold text-warning"
                        : "text-secondary-fixed-dim"
                  }
                >
                  {selectedStats.severity === "target" ? "PREDICTED TARGET" : selectedStats.severity === "suspicious" ? "SUSPICIOUS" : "NORMAL"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-on-surface-variant">FLAGGED FLOWS:</span>
                <DataReadout>{selectedStats.touchingFlows}</DataReadout>
              </div>
              <div className="flex justify-between">
                <span className="text-on-surface-variant">TOTAL BYTES:</span>
                <DataReadout>{selectedStats.totalBytes.toLocaleString()}</DataReadout>
              </div>
              <div className="flex justify-between">
                <span className="text-on-surface-variant">FLOW COUNT:</span>
                <DataReadout>{selectedStats.totalFlowCount.toLocaleString()}</DataReadout>
              </div>
              <div className="flex justify-between border-t border-outline-variant/60 pt-1">
                <span className="text-on-surface-variant">MAX ATTENTION:</span>
                <DataReadout className="text-primary-container">{selectedStats.maxAttentionWeight.toFixed(4)}</DataReadout>
              </div>
            </div>
          </div>
        )}

        {nodes.length === 0 ? (
          <div className="flex h-full items-center justify-center font-mono text-[12px] text-on-surface-variant">
            no flow data
          </div>
        ) : (
          <ForceGraph2D
            ref={graphRef}
            graphData={{ nodes, links }}
            backgroundColor="transparent"
            nodeRelSize={5}
            nodeLabel={(n) => (n as GraphNode).id}
            onNodeClick={(n) => setSelected((n as GraphNode).id)}
            linkDirectionalParticles={(l) => ((l as GraphLink).isPath ? 2 : 0)}
            linkDirectionalParticleColor={() => COLOR_TARGET}
            linkDirectionalParticleWidth={2}
            linkCanvasObjectMode={() => "replace"}
            linkCanvasObject={(l, ctx) => {
              const link = l as GraphLink
              const source = link.source as unknown as GraphNode
              const target = link.target as unknown as GraphNode
              if (source.x === undefined || target.x === undefined) return
              ctx.beginPath()
              ctx.moveTo(source.x, source.y!)
              ctx.lineTo(target.x, target.y!)
              if (link.isPath) {
                // The predicted attack path -- distinct dashed treatment,
                // pulsing opacity, separate from ordinary flow edges.
                ctx.save()
                ctx.setLineDash([6, 4])
                ctx.strokeStyle = COLOR_TARGET
                ctx.globalAlpha = pulse
                ctx.lineWidth = 1.5 + link.weight * 3
                ctx.stroke()
                ctx.restore()
              } else {
                ctx.strokeStyle = COLOR_LINK
                ctx.lineWidth = 0.6 + link.weight * 3
                ctx.stroke()
              }
            }}
            nodeCanvasObjectMode={() => "replace"}
            nodeCanvasObject={(n, ctx) => {
              const node = n as GraphNode
              if (node.x === undefined || node.y === undefined) return
              const r = node.id === selected ? 7 : 5
              const color = node.severity === "target" ? COLOR_TARGET : node.severity === "suspicious" ? COLOR_SUSPICIOUS : COLOR_NODE

              // Pulsing glow halo on the predicted target only -- the one
              // element that should be impossible to miss even without
              // reading the banner.
              if (node.severity === "target") {
                ctx.beginPath()
                ctx.arc(node.x, node.y, r + 6 + pulse * 4, 0, 2 * Math.PI)
                ctx.fillStyle = COLOR_TARGET
                ctx.globalAlpha = 0.18 * pulse
                ctx.fill()
                ctx.globalAlpha = 1
              }

              ctx.beginPath()
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
              ctx.fillStyle = color
              ctx.fill()
              if (node.severity !== "normal" || node.id === selected) {
                ctx.beginPath()
                ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI)
                ctx.strokeStyle = node.id === selected ? COLOR_SELECTED_RING : color
                ctx.lineWidth = node.severity === "target" ? 1.5 : 1
                ctx.stroke()
              }
            }}
            d3VelocityDecay={0.35}
            cooldownTicks={100}
            onEngineStop={handleEngineStop}
            width={size.width || undefined}
            height={size.height || undefined}
          />
        )}
      </div>
    </Panel>
  )
}
