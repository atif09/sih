import type { ReactNode } from "react"
import { useInView } from "../../lib/useInView"

/**
 * The page's one scroll-triggered moment, applied consistently everywhere
 * it's used rather than a different effect per section -- content sits
 * slightly below its resting place and fades up into it the first time it
 * enters view. `delayMs` staggers a genuine sequence (the numbered
 * workflow steps, the stage cards); everything else reveals as one block,
 * not card-by-card, to keep this restrained rather than scattered.
 * `prefers-reduced-motion` disables the transition in CSS (index.css),
 * not here -- content is already in the DOM either way, just instant.
 */
export function Reveal({ children, delayMs = 0, className = "" }: { children: ReactNode; delayMs?: number; className?: string }) {
  const { ref, inView } = useInView<HTMLDivElement>()

  return (
    <div
      ref={ref}
      className={`reveal ${inView ? "reveal-visible" : ""} ${className}`}
      style={{ transitionDelay: inView ? `${delayMs}ms` : "0ms" }}
    >
      {children}
    </div>
  )
}
