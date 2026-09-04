/**
 * Maps prediction data to the three functional signal roles this app needs
 * (info/warning/critical). Colors are never decorative -- each mapping
 * encodes a real severity judgment, and every color used is one of the
 * named tokens in index.css, never a one-off hex chosen per component.
 */

export type Signal = "info" | "warning" | "critical"

export function probabilitySignal(p: number): Signal {
  if (p >= 0.75) return "critical"
  if (p >= 0.5) return "warning"
  return "info"
}

export const SIGNAL_HEX: Record<Signal, string> = {
  info: "#4edea3", // secondary-fixed-dim
  warning: "#f59e0b", // warning (see index.css's note on this one exception)
  critical: "#ffb4ab", // error
}

export const SIGNAL_TEXT_CLASS: Record<Signal, string> = {
  info: "text-secondary-fixed-dim",
  warning: "text-warning",
  critical: "text-error",
}

export const SIGNAL_BORDER_CLASS: Record<Signal, string> = {
  info: "border-secondary-fixed-dim",
  warning: "border-warning",
  critical: "border-error",
}

export const SIGNAL_BG_CLASS: Record<Signal, string> = {
  info: "bg-secondary-fixed-dim",
  warning: "bg-warning",
  critical: "bg-error",
}

/** Stage basis ("benign" always reads as info regardless of the raw
 * probability that produced it -- the stage classification is the more
 * meaningful severity signal than the bare number once a stage exists). */
export function stageSignal(stage: { stage: string; basis: string }): Signal {
  if (stage.basis === "benign") return "info"
  if (stage.stage === "Denial of Service / Impact") return "critical"
  return "warning"
}

/** Momentum (Phase 9, Feature 1) is a 0-100 rate-of-escalation score, not a
 * probability -- thresholds chosen independently of probabilitySignal's. */
export function momentumSignal(score: number): Signal {
  if (score >= 60) return "critical"
  if (score >= 30) return "warning"
  return "info"
}
