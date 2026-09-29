import { useEffect, useRef, useState } from "react"

/**
 * Fires once when the element first enters the viewport, then disconnects
 * -- a one-shot reveal trigger, not a scroll-position tracker. No library:
 * IntersectionObserver covers this exactly, and this page has no other use
 * for scroll math.
 */
export function useInView<T extends HTMLElement>(threshold = 0.15) {
  const ref = useRef<T | null>(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true)
          observer.disconnect()
        }
      },
      { threshold },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [threshold])

  return { ref, inView }
}
