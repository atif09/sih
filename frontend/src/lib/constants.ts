/**
 * Values that must mirror the backend exactly, not be re-derived here.
 */

// Mirrors surprise/score.py's ALARM_PERCENTILE. The backend already applies
// this threshold to compute `surprise_alarm` per window -- this constant is
// ONLY for drawing the reference line at the same value, never for deciding
// alarm state client-side (that decision always comes from the API).
export const ALARM_PERCENTILE = 97.5
