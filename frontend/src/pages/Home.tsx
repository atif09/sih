import { ArrowRight, Search } from "lucide-react"
import { FloatingArtifacts } from "../components/home/FloatingArtifacts"
import { Reveal } from "../components/home/Reveal"
import { IconExplainability, IconRadar, IconTopology, IconUpload } from "../lib/icons"
import { Badge } from "../components/primitives/Badge"
import type { Signal } from "../lib/severity"

/**
 * Marketing/landing page, following the reference "MoleculeAI" screenshot's
 * structure and typographic treatment closely (huge two-line headline with
 * a dimmed second line, monospace command-bar hero control, stat row,
 * 4-step pipeline, a feature list + sample-data card, a target-style card
 * grid, tech-stack row, closing CTA with a code snippet). Reskinned onto
 * this project's own console tokens/type for brand continuity, and every
 * number/fact on this page is real -- pulled from the project's own
 * validated benchmark results and real API field names, not placeholders.
 */

const STATS = [
  { value: "0.989", label: "AUC on unseen attack type" },
  { value: "0.933", label: "world model F1 (baseline: 0.000)" },
  { value: "6", label: "MITRE stages mapped" },
]

const WORKFLOW = [
  {
    n: "01",
    icon: IconUpload,
    title: "Ingest",
    body: "Flow CSV or raw PCAP, parsed through the same CICFlowMeter pipeline used to build the training data.",
  },
  {
    n: "02",
    icon: IconTopology,
    title: "Construct",
    body: "Hosts become graph nodes, aggregated flows become edges -- one graph snapshot per time window.",
  },
  {
    n: "03",
    icon: IconRadar,
    title: "Forecast",
    body: "A GNN + temporal world model rolls the graph state forward K steps, entirely from its own predictions.",
  },
  {
    n: "04",
    icon: IconExplainability,
    title: "Explain",
    body: "Real GNN attention and gradient-based saliency attach to every prediction, mapped to a MITRE ATT&CK stage.",
  },
]

const PROFILE_ITEMS = [
  "Host & flow graph construction",
  "K-step infiltration forecasting",
  "MITRE ATT&CK stage mapping",
  "Cyber Threat Momentum scoring",
  "Containment window estimation",
  "Counterfactual intervention simulation",
]

const STAGES: { tag: string; signal: Signal; code: string; title: string; body: string }[] = [
  { tag: "RECON", signal: "info", code: "PortScan", title: "Reconnaissance", body: "Port and host discovery before an intrusion begins." },
  { tag: "ACCESS", signal: "warning", code: "Web Attack, Patator", title: "Initial Access", body: "Credential brute-forcing or exploiting a public-facing app to get in." },
  { tag: "LATERAL", signal: "warning", code: "Infiltration", title: "Lateral Movement", body: "Post-compromise pivoting against other internal hosts." },
  { tag: "C2", signal: "critical", code: "Bot", title: "Command and Control", body: "Botnet beaconing back to an attacker-controlled server." },
  { tag: "EXFIL", signal: "critical", code: "heuristic fallback only", title: "Exfiltration", body: "No CIC-IDS2017 category maps cleanly here -- reachable only through the raw-feature heuristic path." },
  { tag: "IMPACT", signal: "critical", code: "DoS, Heartbleed", title: "Denial of Service / Impact", body: "Volumetric or crash-style attacks that don't fit the 5-stage kill chain -- an honest 6th label, not a forced fit." },
]

const STACK = ["PyTorch", "PyTorch Geometric", "FastAPI", "React", "scikit-learn", "CICFlowMeter"]

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" })
}

export function Home({ onLaunch }: { onLaunch: () => void }) {
  return (
    <div className="home-page min-h-screen bg-surface-container-lowest font-sans text-on-surface">
      <nav className="sticky top-0 z-50 flex items-center justify-between border-b border-outline-variant bg-surface-container-lowest/95 px-6 py-4 backdrop-blur">
        <span className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <span className="h-2 w-2 rounded-full bg-primary-container" />
          NeuralShield
        </span>
        <div className="hidden items-center gap-8 font-mono text-[12px] text-on-surface-variant md:flex">
          <button onClick={() => scrollTo("workflow")} className="transition-colors hover:text-on-surface">
            How it works
          </button>
          <button onClick={() => scrollTo("stages")} className="transition-colors hover:text-on-surface">
            Stages
          </button>
          <button onClick={() => scrollTo("profiling")} className="transition-colors hover:text-on-surface">
            Insights
          </button>
        </div>
        <button
          onClick={onLaunch}
          className="bg-on-surface px-4 py-2 font-mono text-[12px] font-semibold uppercase tracking-wider text-surface-container-lowest transition-opacity hover:opacity-90"
        >
          Launch Console
        </button>
      </nav>

      {/* ---------------------------------------------------------------- HERO */}
      <section className="relative overflow-hidden border-b border-outline-variant px-6 py-28 text-center">
        <FloatingArtifacts />
        <div className="relative mx-auto flex max-w-3xl flex-col items-center gap-6">
          <span className="border border-outline-variant bg-surface-container px-3 py-1 font-mono text-[10px] uppercase tracking-[0.2em] text-on-surface-variant">
            AI-powered attack forecasting
          </span>
          <h1 className="text-5xl font-bold leading-[1.05] tracking-tight sm:text-6xl">
            <span className="text-on-surface">Forecast any attack.</span>
            <br />
            <span className="text-on-surface-variant">Instantly.</span>
          </h1>
          <p className="max-w-xl text-[15px] leading-relaxed text-on-surface-variant">
            From raw flow data to K-step infiltration forecasts, MITRE stage mapping, and defender simulation.
            Precision-engineered for SOC analysts, not dashboards.
          </p>

          <div className="flex w-full max-w-lg flex-col items-center gap-3 sm:flex-row">
            <button
              onClick={onLaunch}
              className="flex w-full flex-1 items-center gap-2 border border-outline-variant bg-surface-container px-4 py-3 text-left font-mono text-[13px] text-on-surface-variant transition-colors hover:border-primary-container"
            >
              <Search size={15} className="shrink-0" />
              Try: a CIC-IDS2017 capture, or drop your own PCAP...
            </button>
            <button
              onClick={onLaunch}
              className="w-full shrink-0 bg-on-surface px-5 py-3 font-mono text-[13px] font-bold uppercase tracking-wider text-surface-container-lowest transition-opacity hover:opacity-90 sm:w-auto"
            >
              Analyze
            </button>
          </div>

          <button
            onClick={() => scrollTo("workflow")}
            className="mt-2 border border-outline-variant px-4 py-2 font-mono text-[11px] text-on-surface-variant transition-colors hover:text-on-surface"
          >
            See how it works
          </button>
        </div>
      </section>

      {/* --------------------------------------------------------------- STATS */}
      <Reveal className="grid grid-cols-1 divide-x divide-y divide-outline-variant border-b border-outline-variant sm:grid-cols-3 sm:divide-y-0">
        {STATS.map((s) => (
          <div key={s.label} className="flex flex-col gap-1 px-6 py-8 text-center">
            <DataStat value={s.value} />
            <span className="font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">{s.label}</span>
          </div>
        ))}
      </Reveal>

      {/* ------------------------------------------------------------ WORKFLOW */}
      <section id="workflow" className="border-b border-outline-variant px-6 py-24">
        <p className="mb-12 text-center font-mono text-[10px] uppercase tracking-[0.25em] text-on-surface-variant">Forecast pipeline</p>
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-4">
          {WORKFLOW.map((step, i) => (
            <Reveal key={step.n} delayMs={i * 90} className="flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-on-surface-variant/40">{step.n}</span>
                <span className="flex h-8 w-8 items-center justify-center border border-outline-variant text-primary-container">
                  <step.icon size={15} />
                </span>
              </div>
              <h3 className="text-[15px] font-semibold">{step.title}</h3>
              <p className="text-[13px] leading-relaxed text-on-surface-variant">{step.body}</p>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------ PROFILE */}
      <section id="profiling" className="border-b border-outline-variant px-6 py-24">
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-16 lg:grid-cols-2">
          <Reveal>
            <h2 className="mb-8 text-3xl font-bold tracking-tight">Full-Spectrum Attack Profiling</h2>
            <ol className="divide-y divide-outline-variant border-t border-outline-variant">
              {PROFILE_ITEMS.map((item, i) => (
                <li key={item} className="flex items-center gap-4 py-3">
                  <span className="font-mono text-[11px] text-on-surface-variant/50">{String(i + 1).padStart(2, "0")}</span>
                  <span className="text-[14px] text-on-surface">{item}</span>
                </li>
              ))}
            </ol>
          </Reveal>

          <Reveal delayMs={120} className="border border-outline-variant bg-surface-container-low p-6">
            <div className="mb-6 flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">Sample analysis</span>
              <Badge signal="warning">window 30</Badge>
            </div>
            <h3 className="mb-6 font-mono text-lg font-semibold">Friday-Afternoon-DDoS</h3>

            <div className="mb-6 border border-outline-variant bg-surface-container-lowest p-8 text-center">
              <div className="font-mono text-5xl font-bold text-primary-container">0.540</div>
              <div className="mt-2 font-mono text-[10px] uppercase tracking-wide text-on-surface-variant">infiltration probability</div>
            </div>

            <div className="grid grid-cols-2 gap-4 font-mono text-[12px]">
              <Stat label="stage" value="Reconnaissance" />
              <Stat label="basis" value="heuristic_fallback" />
              <Stat label="flow count" value="4,287" />
              <Stat label="attention" value="0.6250" />
            </div>
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------------------- STAGES */}
      <section id="stages" className="border-b border-outline-variant px-6 py-24">
        <div className="mx-auto max-w-5xl">
          <Reveal className="mb-10 flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-3xl font-bold tracking-tight">Core MITRE Stages</h2>
              <p className="mt-2 text-[13px] text-on-surface-variant">The kill-chain vocabulary the world model actually predicts.</p>
            </div>
          </Reveal>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {STAGES.map((s, i) => (
              <Reveal key={s.title} delayMs={i * 60} className="flex flex-col gap-3 border border-outline-variant bg-surface-container-low p-5">
                <div className="flex items-center justify-between">
                  <Badge signal={s.signal}>{s.tag}</Badge>
                  <span className="font-mono text-[10px] text-on-surface-variant">{s.code}</span>
                </div>
                <h3 className="text-[15px] font-semibold">{s.title}</h3>
                <p className="flex-1 text-[12px] leading-relaxed text-on-surface-variant">{s.body}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------------- STACK */}
      <Reveal className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4 border-b border-outline-variant px-6 py-14">
        {STACK.map((t) => (
          <span key={t} className="font-mono text-[11px] uppercase tracking-wider text-on-surface-variant/60">
            {t}
          </span>
        ))}
      </Reveal>

      {/* ------------------------------------------------------------------ CTA */}
      <Reveal className="flex flex-col items-center gap-5 px-6 py-28 text-center">
        <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">Ready to forecast an attack?</h2>
        <p className="max-w-md text-[14px] text-on-surface-variant">
          Launch the console, upload a capture, and watch the world model roll the network state forward.
        </p>
        <button
          onClick={onLaunch}
          className="flex items-center gap-2 bg-on-surface px-6 py-3 font-mono text-[13px] font-bold uppercase tracking-wider text-surface-container-lowest transition-opacity hover:opacity-90"
        >
          Launch Console <ArrowRight size={14} />
        </button>
        <code className="mt-2 font-mono text-[11px] text-on-surface-variant/60">POST /analyze -F csv_file=@capture.csv</code>
      </Reveal>

      {/* --------------------------------------------------------------- FOOTER */}
      <footer className="flex flex-col gap-6 border-t border-outline-variant px-6 py-10 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <span className="flex items-center gap-2 text-sm font-semibold tracking-tight">
            <span className="h-2 w-2 rounded-full bg-primary-container" />
            NeuralShield
          </span>
          <p className="mt-1 font-mono text-[11px] text-on-surface-variant">GNN + temporal world model for network attack forecasting.</p>
        </div>
        <div className="font-mono text-[11px] text-on-surface-variant">
          <button onClick={onLaunch} className="transition-colors hover:text-on-surface">
            Console
          </button>
        </div>
      </footer>
    </div>
  )
}

function DataStat({ value }: { value: string }) {
  return <div className="font-mono text-3xl font-bold text-on-surface">{value}</div>
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1 border-t border-outline-variant pt-2">
      <span className="uppercase tracking-wide text-on-surface-variant">{label}</span>
      <span className="text-on-surface">{value}</span>
    </div>
  )
}
