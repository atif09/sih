/**
 * Mono-set numeric/technical readout -- JetBrains Mono, used only where
 * digit alignment or raw-data precision is actually meaningful
 * (probabilities, IPs, ports, feature scores, timestamps). Never used as a
 * generic label font -- Geist Sans carries UI chrome/prose.
 */
export function DataReadout({
  children,
  className = "",
  dim = false,
}: {
  children: React.ReactNode
  className?: string
  dim?: boolean
}) {
  return (
    <span className={`font-mono tabular-nums ${dim ? "text-on-surface-variant" : "text-on-surface"} ${className}`}>
      {children}
    </span>
  )
}
