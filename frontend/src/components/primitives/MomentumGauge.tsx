import { momentumSignal, SIGNAL_HEX } from "../../lib/severity"

const CX = 100
const CY = 96
const RADIUS = 78
const STROKE = 10

function angleForValue(value: number, max: number): number {
  return 180 - (value / max) * 180
}

function pointAt(angleDeg: number, radius: number) {
  const rad = (angleDeg * Math.PI) / 180
  return { x: CX + radius * Math.cos(rad), y: CY - radius * Math.sin(rad) }
}

function arcPath(fromValue: number, toValue: number, max: number, radius: number): string {
  const p1 = pointAt(angleForValue(fromValue, max), radius)
  const p2 = pointAt(angleForValue(toValue, max), radius)
  return `M ${p1.x} ${p1.y} A ${radius} ${radius} 0 0 1 ${p2.x} ${p2.y}`
}

/**
 * Semicircular instrument dial for the Momentum Score -- kept as a real
 * dial (needle + tick marks), matching the Stitch mockup's own gauge
 * concept, just re-derived in our token colors instead of the mockup's
 * one-off SVG gradient. Band thresholds match `momentumSignal` exactly.
 */
export function MomentumGauge({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(100, score))
  const needleAngle = angleForValue(clamped, 100)
  const signal = momentumSignal(clamped)

  return (
    <svg viewBox="0 -8 200 114" className="w-full max-w-60" role="img" aria-label={`momentum score ${clamped.toFixed(1)} of 100`}>
      <path d={arcPath(0, 30, 100, RADIUS)} stroke={SIGNAL_HEX.info} strokeWidth={STROKE} fill="none" opacity={0.85} />
      <path d={arcPath(30, 60, 100, RADIUS)} stroke={SIGNAL_HEX.warning} strokeWidth={STROKE} fill="none" opacity={0.85} />
      <path d={arcPath(60, 100, 100, RADIUS)} stroke={SIGNAL_HEX.critical} strokeWidth={STROKE} fill="none" opacity={0.85} />

      {[0, 25, 50, 75, 100].map((tick) => {
        const p = pointAt(angleForValue(tick, 100), RADIUS + 15)
        return (
          <text key={tick} x={p.x} y={p.y} textAnchor="middle" dominantBaseline="middle" className="fill-on-surface-variant font-mono" fontSize={9}>
            {tick}
          </text>
        )
      })}

      <g
        style={{
          transform: `rotate(${90 - needleAngle}deg)`,
          transformOrigin: `${CX}px ${CY}px`,
          transition: "transform 700ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}
      >
        <line x1={CX} y1={CY} x2={CX} y2={CY - RADIUS + 16} stroke={SIGNAL_HEX[signal]} strokeWidth={2} />
      </g>
      <circle cx={CX} cy={CY} r={4.5} fill={SIGNAL_HEX[signal]} />
    </svg>
  )
}
