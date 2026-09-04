// d3-force-3d doesn't ship its own TypeScript declarations. Only forceX/forceY
// are used in this project (TopologyGraph.tsx), typed minimally against the
// same shape as d3-force's forceX/forceY (which this package is a superset
// of, adding an unused-here z dimension).
declare module "d3-force-3d" {
  export interface ForceXY<NodeDatum> {
    (alpha: number): void
    initialize?(nodes: NodeDatum[]): void
    strength(strength: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): ForceXY<NodeDatum>
    x(x: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): ForceXY<NodeDatum>
    y(y: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): ForceXY<NodeDatum>
  }
  export function forceX<NodeDatum = unknown>(x?: number): ForceXY<NodeDatum>
  export function forceY<NodeDatum = unknown>(y?: number): ForceXY<NodeDatum>
}
