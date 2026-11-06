# NeuralShield — AI-Based Network Attack Forecasting

**SIH 2026 · PS 26153 · National Technical Research Organisation (NTRO)**

A world model that learns how a network's state evolves over time and
forecasts attacker,progression *before* compromise completes — not a
classifier that labels traffic after the fact.

---

## 1. The problem

Conventional intrusion detection looks at one flow or one time window in
isolation and outputs a benign/malicious label. It has no memory and no
sense of direction — it cannot tell a host that has been quietly scanning
for five minutes from a host that has, always talked to many peers, because
it only ever sees the current instant.

PS 26153 asks for something different: a system that models the network as
an evolving state, predicts where that state is heading, and flags an
attack in progress before the kill chain completes.

## 2. Our approach

Each 60-second window of traffic becomes a **graph** — hosts are nodes,
aggregated host-pair traffic is edges. A Graph Neural Network encodes that
graph into a latent vector; a temporal model (LSTM, swappable for a
Transformer) consumes the last 20 latents and predicts the **next** one,
before it is ever observed. The attack-forecast head,reads that predicted
future latent, not the true one — this is what makes the output a genuine
forecast rather than a delayed classification.

Because the model predicts a latent instead of raw traffic, it can roll
forward K steps by feeding its own predictions back into itself, with no
future ground,truth required — the same principle behind world models in
Ha & Schmidhuber (2018) and Hafner et al. (2020).

## 3. Headline result

Evaluated on **Friday-Afternoon-DDoS**, a CIC-IDS2017 capture held out
*entirely* from training, validation, and normalization — the model never
saw this attack type during training.

| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| **World model** | 0.963 | 0.951 | 0.917 | **0.933** | 0.019 | **0.989** |
| Logistic-regression baseline (same pooled features, no temporal context) | 0.696 | 0.000 | 0.000 | 0.000 | 0.099 | 0.040 |

This is one unseen attack type on one dataset, not a general
"works-on-any-network" claim.

## 4. What's built

**Core pipeline**
- Flow-level + packet-level feature extraction (CICFlowMeter-schema CSV, or raw PCAP via Scapy)
- Graph construction — host nodes, directed aggregated-flow edges, per 60s window
- GNN (TransformerConv) + temporal core (LSTM/Transformer, swappable) world model, ~101K parameters
- K-step forward rollout from a live context window
- Mapping from predicted state to a MITRE ATT&CK stage (Reconnaissance → Initial Access → Lateral Movement → Command & Control → Exfiltration, plus an honest 6th "Denial of Service / Impact" label for attacks that don't fit the five-stage PS framing)
- Explainability: real GNN edge attention + Input×Gradient feature attribution — not a SHAP wrapper, not a stub

**Analyst-facing differentiators**, each a real computation on the model's own outputs, not a separate trained model:

| Feature | What it answers |
|---|---|
| Cyber Threat Momentum Score | Is risk escalating, and how fast? |
| Containment Window Estimator | How long does this stage typically last, from real historical episodes? |
| Counterfactual Intervention Simulator | What happens to the forecast if I isolate this host / block this flow? |
| Response Planner | Which of every candidate defender action reduces risk the most? |
| Attack Future Replay | Scrub through observed + forecast windows with real per-window attention |
| Multiple Parallel Futures | The model's own uncertainty across plausible outcomes (MC-Dropout) |
| Time-to-Compromise | Cumulative probability of crossing the attack threshold by minute k |
| Explainable Attack Fingerprint | Current vs. predicted similarity to each MITRE stage |
| Surprise Alarm | Novelty detection from the model's own dynamics-prediction error — flags behavior unlike anything in training, without needing an attack-type label |
| Network Forecast View | Decodes the predicted future latent into real, human-readable stats (flow count, port concentration, bytes) |

**Console**: React 19 / TypeScript / Tailwind CSS 4 — three modes (Live Feed, Response Center, Attack Replay), fully offline inference, no cloud API calls.

## 5. Architecture

```
Upload (CSV / PCAP)
      │
      ▼
Feature extraction  ── flow-level (CICFlowMeter schema) + packet-level (Scapy)
      │
      ▼
Graph construction   ── one host graph per 60s window
      │
      ▼
GNN encoder (TransformerConv) ──► latent z_t
      │
      ▼
Temporal core (LSTM / Transformer) ──► predicts ẑ_{t+1}
      │
      ├──► Attack-forecast head      → infiltration probability, K-step rollout
      ├──► Attack-type head          → MITRE stage mapping
      ├──► Dynamics error            → Surprise Alarm
      └──► Forecast decoder (frozen-backbone probe) → Network Forecast View
      │
      ▼
Explainability (attention + gradient attribution) + analyst features
      │
      ▼
FastAPI  ──►  React console
```

### Tech stack

| Layer | Technologies |
|---|---|
| Core ML | Python 3.12, PyTorch, PyTorch Geometric |
| Data | pandas, NumPy, scikit-learn |
| Traffic processing | Scapy, cicflowmeter |
| API | FastAPI, Uvicorn |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS 4, Recharts |
| Deployment | Docker (single container serves both API and built frontend), Hugging Face Spaces |

## 6. Setup & running it

Requires Python 3.12 and Node 18+.

```bash
# Backend
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn api.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # proxies /api -> localhost:8000, served at localhost:5173
```

Then open `http://localhost:5173`, upload a CICFlowMeter-style CSV or a raw
PCAP, and the console runs inference live. Everything after model load is
fully offline — no external API calls.

### API reference

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /model-info` | Liveness, model metadata |
| `POST /analyze` | CSV/PCAP + K → timeline, forecast, MITRE stage, explanation, momentum, containment, parallel futures, fingerprint, surprise alarm, time-to-compromise, network forecast |
| `POST /simulate` | Counterfactual action on the last analyzed context |
| `POST /plan-response` | Ranks every candidate defender action by measured risk reduction |
| `GET /model-performance/roc-pr`, `/calibration` | Precomputed held-out evaluation data |

## 7. Roadmap

1. Real packet-level training signal (raw PCAP already downloadable from UNB; extraction code is built and tested, retraining is the remaining step)
2. Second dataset for genuine cross-dataset generalization evidence (CTU-13 scoped, not yet integrated)
3. Temperature scaling for calibration.
4. Live NetFlow/IPFIX ingestion, auth, multi-session state

## 8. Project layout

```
features/     Flow-level + packet-level extraction, windowing
graph/        Graph state representation, ground-truth transitions
models/       World model (GNN + temporal core), forecast decoder
rollout/      K-step rollout engine, MITRE stage mapping
explain/      Attention + gradient-based explainability
momentum/     Cyber Threat Momentum Score
containment/  Containment Window Estimator, Response Planner
futures/      Multiple Parallel Futures, Time-to-Compromise
fingerprint/  Explainable Attack Fingerprint
surprise/     Surprise Alarm
baseline/     Logistic-regression baseline
benchmark/    World model vs. baseline comparison
api/          FastAPI inference service
frontend/     React console
scripts/      Training, benchmarking, and data-export utilities
configs/      Reproducible run configuration
tests/        ~196 tests across 21 files
```
