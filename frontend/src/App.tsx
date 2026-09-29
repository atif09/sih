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
import { SurpriseAlarmPanel } from "./components/panels/SurpriseAlarmPanel"
import { NetworkForecastCard } from "./components/panels/NetworkForecastCard"
import { TimeToCompromisePanel } from "./components/panels/TimeToCompromisePanel"
import { ParallelFuturesTree } from "./components/panels/ParallelFuturesTree"
import { ResponsePlannerPanel } from "./components/panels/ResponsePlannerPanel"
import { ReplayPanel } from "./components/panels/ReplayPanel"
import { IconSpotlight } from "./lib/icons"
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
  // Dims every current-state panel (kill chain, topology, explainability),
  // leaving only the Trajectory Timeline's forecast highlighted -- a
  // one-click way to answer "show me just the prediction" live, since that
  // question came up directly (current-state detection and future
  // forecasting sit right next to each other on this page and are easy to
  // conflate without pointing at exactly the right pixel).
  const [predictionSpotlight, setPredictionSpotlight] = useState(false)

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
                <div className="flex justify-end">
                  <button
                    onClick={() => setPredictionSpotlight((v) => !v)}
                    className={`flex items-center gap-1.5 border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wide transition-colors ${
                      predictionSpotlight
                        ? "border-error bg-error-container/10 text-error"
                        : "border-outline-variant text-on-surface-variant hover:text-on-surface"
                    }`}
                  >
                    <IconSpotlight size={13} />
                    {predictionSpotlight ? "exit prediction spotlight" : "show prediction only"}
                  </button>
                </div>

                <div className={predictionSpotlight ? "pointer-events-none opacity-20 grayscale transition-all duration-300" : "transition-all duration-300"}>
                  <div className="flex flex-col gap-3">
                    <KillChainTracker result={result} />
                    <TopologyGraph flows={result.flagged_flows} currentStage={result.forecast[0]?.mitre_stage} />
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
                  <div className={`lg:col-span-7 transition-all duration-300 ${predictionSpotlight ? "ring-2 ring-error" : ""}`}>
                    <TrajectoryTimeline timeline={result.timeline} forecast={result.forecast} />
                  </div>
                  <div
                    className={`lg:col-span-5 ${predictionSpotlight ? "pointer-events-none opacity-20 grayscale transition-all duration-300" : "transition-all duration-300"}`}
                  >
                    <ExplainabilityPanel explanation={result.explanation} />
                  </div>
                </div>

                <div className={`transition-all duration-300 ${predictionSpotlight ? "ring-2 ring-error" : ""}`}>
                  <SurpriseAlarmPanel timeline={result.timeline} summary={result.surprise_alarm} />
                </div>

                <div className={`transition-all duration-300 ${predictionSpotlight ? "ring-2 ring-error" : ""}`}>
                  <NetworkForecastCard forecast={result.network_forecast} />
                </div>

                <div className={predictionSpotlight ? "pointer-events-none opacity-20 grayscale transition-all duration-300" : "transition-all duration-300"}>
                  <AttackFingerprintPanel fingerprint={result.attack_fingerprint} />
                </div>
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
                <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
                  <div className="lg:col-span-8">
                    <CounterfactualPanel flaggedFlows={result.flagged_flows} defaultK={Math.min(5, result.forecast.length || 5)} />
                  </div>
                  <div className="flex flex-col gap-3 lg:col-span-4">
                    <MomentumPanel momentum={result.momentum} currentProbability={result.forecast[0]?.infiltration_probability} />
                    <TimeToCompromisePanel ttc={result.time_to_compromise} />
                  </div>
                </div>
                <ResponsePlannerPanel defaultK={Math.min(5, result.forecast.length || 5)} />
                <ParallelFuturesTree futures={result.multiple_futures} />
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
