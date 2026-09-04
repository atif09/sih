import { useEffect, useRef, useState } from "react"
import { IconUpload } from "../../lib/icons"
import { analyze } from "../../lib/api"
import type { AnalyzeResponse } from "../../lib/types"

type Status = "idle" | "analyzing" | "error"

const PROCESSING_STEPS = [
  "reading capture",
  "extracting flow features",
  "constructing graph state",
  "running world model inference",
  "rolling out k-step forecast",
  "mapping mitre att&ck stages",
  "computing attention + saliency",
]

function fileKind(name: string): "csv" | "pcap" | null {
  const lower = name.toLowerCase()
  if (lower.endsWith(".csv")) return "csv"
  if (lower.endsWith(".pcap") || lower.endsWith(".pcapng")) return "pcap"
  return null
}

/**
 * Loading state is a console-style step log with a blinking cursor, not a
 * generic spinner. Steps are a plausible fixed sequence (the backend is one
 * synchronous POST, no real progress events) advancing on a timer while the
 * request is in flight; settles immediately once the real response
 * arrives, so it never claims to be further along than the actual request.
 */
export function UploadPanel({
  onResult,
  k,
  onKChange,
}: {
  onResult: (r: AnalyzeResponse, filename: string) => void
  k: number
  onKChange: (k: number) => void
}) {
  const [status, setStatus] = useState<Status>("idle")
  const [stepIndex, setStepIndex] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    if (status !== "analyzing") return
    setStepIndex(0)
    timerRef.current = window.setInterval(() => {
      setStepIndex((i) => Math.min(i + 1, PROCESSING_STEPS.length - 1))
    }, 500)
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [status])

  async function handleFile(file: File) {
    const kind = fileKind(file.name)
    if (!kind) {
      setStatus("error")
      setError(`unrecognized file type: ${file.name} (expected .csv, .pcap, or .pcapng)`)
      return
    }
    setStatus("analyzing")
    setError(null)
    try {
      const result = await analyze(file, kind, k)
      onResult(result, file.name)
      setStatus("idle")
    } catch (err) {
      setStatus("error")
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail || (err as Error).message || "request failed")
    }
  }

  return (
    <div className="flex items-center gap-3">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          const file = e.dataTransfer.files[0]
          if (file) void handleFile(file)
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex h-8 min-w-64 cursor-pointer items-center gap-2 border px-3 font-mono text-[11px] transition-colors ${
          dragOver ? "border-primary-container bg-primary-container/10" : "border-outline-variant bg-surface-container"
        }`}
      >
        <IconUpload size={13} className="shrink-0 text-on-surface-variant" />
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.pcap,.pcapng"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void handleFile(file)
            e.target.value = ""
          }}
        />

        {status === "idle" && <span className="truncate text-on-surface-variant">drop capture or click to launch analyzer</span>}
        {status === "analyzing" && (
          <div className="flex items-center gap-2 text-secondary-fixed-dim">
            <span className="truncate">{PROCESSING_STEPS[stepIndex]}</span>
            <span className="animate-pulse">_</span>
          </div>
        )}
        {status === "error" && <span className="truncate text-error">{error}</span>}
      </div>

      <label className="flex shrink-0 items-center gap-1.5 font-mono text-[10px] text-on-surface-variant">
        horizon
        <input
          type="number"
          min={1}
          max={50}
          value={k}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => onKChange(Number(e.target.value))}
          className="w-12 border border-outline-variant bg-surface-container-lowest px-1.5 py-1 text-center text-[11px] text-primary focus:border-primary-container focus:outline-none"
        />
        <span>steps</span>
      </label>
    </div>
  )
}
