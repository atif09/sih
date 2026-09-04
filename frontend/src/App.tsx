import { useState } from "react"
import { Header, type Mode } from "./components/layout/Header"
import { Footer } from "./components/layout/Footer"
import { KillChainTracker } from "./components/panels/KillChainTracker"
import { TopologyGraph } from "./components/panels/TopologyGraph"
import { TrajectoryTimeline } from "./components/panels/TrajectoryTimeline"
import { ExplainabilityPanel } from "./components/panels/ExplainabilityPanel"
import { MomentumPanel } from "./components/panels/MomentumPanel"
import { ContainmentBanner } from "./components/panels/ContainmentBanner"
import { CounterfactualPanel } from "./components/panels/CounterfactualPanel"
import { AttackFingerprintPanel } from "./components/panels/AttackFingerprintPanel"
import { ParallelFuturesPanel } from "./components/panels/ParallelFuturesPanel"
import { ReplayPanel } from "./components/panels/ReplayPanel"
import type { AnalyzeResponse } from "./lib/types"

function EmptyState({ children }: { children: string }) {
  return (
    <div className="flex flex-1 items-center justify-center text-center font-mono text-[13px] text-on-surface-variant">
      {children}
    </div>
  )
}

function App({ onHome }: { onHome?: () => void }) {
  const [mode, setMode] = useState<Mode>("live")
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [filename, setFilename] = useState("")
  const [k, setK] = useState(10)

  function handleResult(r: AnalyzeResponse, name: string) {
    setResult(r)
    setFilename(name)
  }

  function handleReset() {
    setResult(null)
    setFilename("")
  }

  return (
    <div className="flex min-h-screen flex-col bg-surface-container-lowest font-sans text-on-surface">
      <Header mode={mode} onModeChange={setMode} result={result} filename={filename} k={k} onKChange={setK} onResult={handleResult} onReset={handleReset} onHome={onHome} />

      <main className="flex flex-1 flex-col overflow-y-auto">
        {mode === "live" && (
          <section className="flex flex-1 flex-col gap-3 p-3">
            {result ? (
              <>
                <KillChainTracker result={result} />
                <TopologyGraph flows={result.flagged_flows} />
                <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
                  <div className="lg:col-span-7">
                    <TrajectoryTimeline timeline={result.timeline} forecast={result.forecast} />
                  </div>
                  <div className="lg:col-span-5">
                    <ExplainabilityPanel explanation={result.explanation} />
                  </div>
                </div>
                <AttackFingerprintPanel fingerprint={result.attack_fingerprint} />
              </>
            ) : (
              <EmptyState>launch the analyzer above to begin live monitoring</EmptyState>
            )}
          </section>
        )}

        {mode === "response" && (
          <section className="flex flex-1 flex-col gap-3 p-3">
            {result ? (
              <>
                <ContainmentBanner containment={result.containment} />
                <div className="grid flex-1 grid-cols-1 gap-3 lg:grid-cols-12">
                  <div className="lg:col-span-8">
                    <CounterfactualPanel flaggedFlows={result.flagged_flows} defaultK={Math.min(5, result.forecast.length || 5)} />
                  </div>
                  <div className="flex flex-col gap-3 lg:col-span-4">
                    <MomentumPanel momentum={result.momentum} />
                    <ParallelFuturesPanel futures={result.multiple_futures} />
                  </div>
                </div>
              </>
            ) : (
              <EmptyState>run an analysis on Mode 1: Live Feed first</EmptyState>
            )}
          </section>
        )}

        {mode === "replay" && (
          <section className="flex flex-1 flex-col gap-3 p-3">
            {result ? <ReplayPanel result={result} /> : <EmptyState>run an analysis on Mode 1: Live Feed first</EmptyState>}
          </section>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default App
