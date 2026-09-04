import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { forceX, forceY } from "d3-force-3d"
import ForceGraph2D, { type ForceGraphMethods, type NodeObject, type LinkObject } from "react-force-graph-2d"
import { IconRadar } from "../../lib/icons"
import { Panel } from "../primitives/Panel"
import { DataReadout } from "../primitives/DataReadout"
import type { FlaggedFlow } from "../../lib/types"

interface GraphNode extends NodeObject {
  id: string
  isEndpointOfAttack: boolean
}
interface GraphLink extends LinkObject {
  source: string
  target: string
  weight: number
  isAttack: boolean
}

const COLOR_NODE = "#00f0ff" // restored to the original cyan-accent node color (matches the line charts' own restored "#00f0ff") -- only the dots, canvas/background stays the light theme
const COLOR_ATTACK = "#ffb4ab" // error -- unchanged
const COLOR_LINK = "#4edea3" // secondary-fixed-dim -- unchanged
const COLOR_SELECTED_RING = "#71717a" // on-surface-variant -- a visible gray halo against the white canvas

interface NodeStats {
  ip: string
  isAttackEndpoint: boolean
  touchingFlows: number
  totalBytes: number
  totalFlowCount: number
  maxAttentionWeight: number
}

function computeNodeStats(ip: string, flows: FlaggedFlow[]): NodeStats {
  const touching = flows.filter((f) => f.src_ip === ip || f.dst_ip === ip)
  return {
    ip,
    isAttackEndpoint: touching.some((f) => f.is_attack_edge),
    touchingFlows: touching.length,
    totalBytes: touching.reduce((a, f) => a + f.bytes_sum, 0),
    totalFlowCount: touching.reduce((a, f) => a + f.flow_count, 0),
    maxAttentionWeight: touching.reduce((a, f) => Math.max(a, f.attention_weight), 0),
  }
}

/**
 * Full-bleed network topology built from `flagged_flows` -- the flows the
 * model's attention actually weighted, not a claim of complete captured
 * topology (the API doesn't expose that). Clicking a node opens a real
 * inspector HUD (the Stitch mockup's node inspector showed fabricated OS
 * fingerprints and degree stats; this one shows only fields the API
 * actually returns, aggregated across that host's flagged flows).
 */
export function TopologyGraph({ flows }: { flows: FlaggedFlow[] }) {
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

  const { nodes, links, hasAttackEdge } = useMemo(() => {
    const nodeIds = new Set<string>()
    const attackEndpoints = new Set<string>()
    for (const f of flows) {
      nodeIds.add(f.src_ip)
      nodeIds.add(f.dst_ip)
      if (f.is_attack_edge) {
        attackEndpoints.add(f.src_ip)
        attackEndpoints.add(f.dst_ip)
      }
    }
    const nodes: GraphNode[] = Array.from(nodeIds).map((id) => ({
      id,
      isEndpointOfAttack: attackEndpoints.has(id),
    }))
    const links: GraphLink[] = flows.map((f) => ({
      source: f.src_ip,
      target: f.dst_ip,
      weight: f.attention_weight,
      isAttack: f.is_attack_edge,
    }))
    return { nodes, links, hasAttackEdge: attackEndpoints.size > 0 }
  }, [flows])

  useEffect(() => {
    graphRef.current?.d3Force("x", forceX(0).strength(0.05))
    graphRef.current?.d3Force("y", forceY(0).strength(0.05))
  }, [nodes, links])

  const handleEngineStop = useCallback(() => {
    graphRef.current?.zoomToFit(400, 56)
  }, [])

  const selectedStats = selected ? computeNodeStats(selected, flows) : null

  return (
    <Panel
      title="Live Attack Topology Graph"
      icon={<IconRadar size={13} className="text-primary-container" />}
      aside={
        <div className="flex items-center gap-3 font-mono text-[10px]">
          <span className="flex items-center gap-1.5 text-on-surface-variant">
            <span className="h-2 w-2 rounded-full bg-error" /> attack vector
          </span>
          <span className="flex items-center gap-1.5 text-on-surface-variant">
            {/* Hardcoded to match COLOR_NODE exactly -- bg-primary-container
                is the shared theme token (black, used app-wide for buttons
                etc.), but the canvas dots themselves are a one-off restored
                cyan accent, not that token, so the legend must match the
                canvas, not the token. */}
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: "#00f0ff" }} /> flow
          </span>
        </div>
      }
      className="reticle-corner flex flex-1 flex-col overflow-hidden"
      bodyClassName="relative flex-1"
    >
      <div
        ref={containerRef}
        className="relative h-full w-full overflow-hidden bg-surface-container-lowest bg-center bg-size-[16px_16px] bg-[radial-gradient(var(--color-outline-variant)_1px,transparent_1px)]"
      >
        <div className="pointer-events-none absolute bottom-2 left-2 z-10 border border-outline-variant bg-surface-container-lowest/80 px-2 py-1 font-mono text-[10px] text-outline">
          [click node to inspect] // {flows.length} flows observed
        </div>
        {hasAttackEdge && (
          <div className="pointer-events-none absolute right-2 top-2 z-10 flex items-center gap-1.5 font-mono text-[10px] text-error">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-error" />
            attack edge active
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
                <span className={selectedStats.isAttackEndpoint ? "font-bold text-error" : "text-secondary-fixed-dim"}>
                  {selectedStats.isAttackEndpoint ? "ATTACK ENDPOINT" : "NO ATTACK FLAG"}
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
            linkDirectionalParticles={(l) => ((l as GraphLink).isAttack ? 2 : 0)}
            linkDirectionalParticleColor={() => COLOR_ATTACK}
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
              ctx.strokeStyle = link.isAttack ? COLOR_ATTACK : COLOR_LINK
              ctx.lineWidth = 0.6 + link.weight * 3
              ctx.stroke()
            }}
            nodeCanvasObjectMode={() => "replace"}
            nodeCanvasObject={(n, ctx) => {
              const node = n as GraphNode
              if (node.x === undefined || node.y === undefined) return
              const r = node.id === selected ? 7 : 5
              ctx.beginPath()
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
              ctx.fillStyle = node.isEndpointOfAttack ? COLOR_ATTACK : COLOR_NODE
              ctx.fill()
              if (node.isEndpointOfAttack || node.id === selected) {
                ctx.beginPath()
                ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI)
                ctx.strokeStyle = node.id === selected ? COLOR_SELECTED_RING : COLOR_ATTACK
                ctx.lineWidth = 1
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
