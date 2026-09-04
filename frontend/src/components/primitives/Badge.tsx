import type { Signal } from "../../lib/severity"
import { SIGNAL_BORDER_CLASS, SIGNAL_TEXT_CLASS } from "../../lib/severity"

/**
 * A small functional status tag -- outlined, not filled, so it reads as
 * data classification rather than a decorative pill. Color always maps to
 * a real Signal, never chosen for variety.
 */
export function Badge({ signal, children }: { signal: Signal; children: React.ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 border px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wide ${SIGNAL_BORDER_CLASS[signal]} ${SIGNAL_TEXT_CLASS[signal]}`}
    >
      {children}
    </span>
  )
}
