import { useEffect, useState } from "react"
import { getHealth, getModelInfo, type HealthResponse, type ModelInfoResponse } from "../../lib/api"

/**
 * Status strip, adapted from the Stitch mockup's footer -- but every value
 * here is a real call to GET /health and GET /model-info, not the
 * mockup's fabricated "LOSS: 0.0142" / checkpoint filename / "ACC: 98.4%"
 * (this API doesn't expose training metrics at all). What's real and
 * genuinely available: model architecture, parameter count, and inference
 * device, straight from the running backend.
 */
export function Footer() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    Promise.all([getHealth(), getModelInfo()])
      .then(([h, m]) => {
        setHealth(h)
        setModelInfo(m)
      })
      .catch(() => setFailed(true))
  }, [])

  return (
    <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-outline-variant bg-surface-container-lowest px-5 py-2 font-mono text-[11px] text-on-surface-variant">
      <div className="flex items-center gap-3">
        {failed ? (
          <span className="flex items-center gap-1.5 text-error">
            <span className="h-1.5 w-1.5 rounded-full bg-error" />
            backend unreachable
          </span>
        ) : health?.status === "ok" ? (
          <span className="flex items-center gap-1.5 font-medium text-secondary-fixed-dim">
            <span className="h-1.5 w-1.5 rounded-full bg-secondary-fixed-dim" />
            GNN inference engine: online
          </span>
        ) : (
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-outline" />
            connecting...
          </span>
        )}
        {modelInfo && (
          <>
            <span className="text-outline-variant">/</span>
            <span>
              temporal core <span className="text-on-surface">{modelInfo.temporal_type}</span>
            </span>
            <span className="text-outline-variant">/</span>
            <span>
              params <span className="text-on-surface">{modelInfo.n_parameters.toLocaleString()}</span>
            </span>
            <span className="text-outline-variant">/</span>
            <span>
              device <span className="text-on-surface">{health?.device ?? "unknown"}</span>
            </span>
          </>
        )}
      </div>
    </footer>
  )
}
