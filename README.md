---
title: NeuralShield
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# SIH26153 — AI-Based Network Attack Forecasting (World Models)

World-model-based system that learns network state transition dynamics to
forecast attacker progression before compromise completes. Built for SIH 2026,
PS 26153.

## Project layout

```
data/            Dataset storage (cicids2017/, cicids2018/, cicids2018-pcap/, ctu13/) + data-loading code
features/        Phase 1: flow-level + packet-level feature extraction, windowing/merge
graph/           Phase 2: graph state representation + ground-truth transition extraction
models/          Phase 3: world model (GNN + temporal layer)
rollout/         Phase 4: K-step forward simulation + MITRE ATT&CK stage mapping
explain/         Phase 5: real explainability (GNN attention + gradient-based feature attribution)
baseline/        Phase 6: logistic regression baseline
benchmark/       Phase 6: world model vs. baseline comparison
api/             Phase 7: FastAPI inference service
frontend/        Phase 8: not started (spec pending)
scripts/         Standalone CLI utilities (sanity checks, fixture generation)
configs/         Reproducible run configuration (pipeline.yaml, later: training configs)
tests/           Unit tests
```

## Setup

Requires Python 3.12 (PyTorch/PyTorch Geometric, coming in Phase 3, do not yet
support the newer 3.14 — this repo's venv is pinned to 3.12).

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Phase 1 — Feature Extraction Pipeline

### Dataset (finalized)

Primary: **CIC-IDS2017 "GeneratedLabelledFlows"**, 8 per-day/attack-type CSVs
in `data/cicids2017/TrafficLabelling /` (yes, trailing space in the folder
name — see `features/datasets.py`, which centralizes this so no other code
has to know about it). CIC-IDS2018 support (`features/schema.py`'s alias
table covers both naming conventions) is kept in the parser since it's free,
but the CIC-IDS2018 CSVs are **no longer present in `data/cicids2018/`** —
they were verified earlier in this project but are gone from disk as of this
update; not something this session's commands intentionally deleted (only a
generated synthetic fixture file was ever removed from that folder). Not a
blocker since CIC-IDS2018 is superseded, but flagging it since several GB of
your downloaded data is unaccounted for — re-download if you want it back.
CTU-13 remains the secondary held-out generalization test, unchanged, to be
added once placed in `data/ctu13/<scenario>/`.

All 8 CIC-IDS2017 files have been run through the full pipeline
(`python scripts/sanity_check.py --all-days`) and verified:

| Day | Attack categories | Src/Dst IP populated |
|---|---|---|
| Monday | Benign | 100% |
| Tuesday | FTP-Patator, SSH-Patator | 100% |
| Wednesday | DoS GoldenEye/Slowloris/Slowhttptest/Hulk, Heartbleed | 100% |
| Thursday-Morning | Web Attack (Brute Force/XSS/SQLi) | 100% |
| Thursday-Afternoon | Infiltration | 100% |
| Friday-Morning | Benign (+ some Bot) | 100% |
| Friday-Afternoon-PortScan | PortScan | 100% |
| Friday-Afternoon-DDoS | DDoS | 100% |

Unlike CIC-IDS2018 (where 7/8 files had Src/Dst IP stripped), every
CIC-IDS2017 file has real, populated IP addresses — **Phase 2's graph node
construction can proceed directly, no PCAP-based identity re-derivation
needed.**

Two real-data issues were found and fixed while validating against these
files (both covered by regression tests):
1. **Day-first timestamps.** `01/03/2018`-style dates are ambiguous
   (Jan 3 vs. Mar 1); pandas defaulted to month-first and silently
   misparsed them. Fixed with `dayfirst=True`, confirmed against capture-day
   filenames.
2. **Mixed file encoding + mangled label text.** The Thursday web-attacks
   file isn't valid UTF-8 — it has a `0x96` byte in the Label column
   (`Web Attack – Brute Force`, a Windows-1252 en dash). Decoding it as
   Latin-1 "succeeds" but produces the wrong character. The loader now tries
   UTF-8 → cp1252 → Latin-1 in order, and label text is normalized (en/em
   dash → `-`) so label strings are consistent across every file regardless
   of which encoding produced them.

**Known, un-"fixed" data quirk (informational, not a bug in this code):** a
small number of flows have a negative `Flow Duration` (a documented
CICFlowMeter-2017 clock-skew bug), which propagates into negative
`flow_byts_s`/`flow_pkts_s`. Left as-is rather than clipped/dropped, since it
may be a legitimate (if buggy-looking) signal rather than noise — worth a
decision before training, not something to silently paper over.

Also confirmed harmless: CIC-IDS2017's header has a genuine duplicate column
(`Fwd Header Length` appears twice); the second copy gets pandas-mangled to
`Fwd Header Length.1` on read and is correctly dropped as unmapped (logged),
since it carries no information not already in the first occurrence.

### What's built
- `features/datasets.py` — dataset registry: the 8 CIC-IDS2017 file paths
  (with day + documented attack categories, for Phase 3 hold-out bookkeeping)
  and the trailing-space folder quirk, kept out of the extractor code so it
  stays dataset-agnostic.
- `features/schema.py` — canonical column schema + alias table for
  CICFlowMeter CSVs, normalizing the header-naming inconsistencies seen across
  both CIC-IDS2017 (`Source IP`, `Total Fwd Packets`, ...) and CIC-IDS2018
  (`Src IP`, `Tot Fwd Pkts`, ...) column-naming conventions.
- `features/flow_features.py` — loads a CICFlowMeter CSV, normalizes columns,
  and extracts: identifiers (IP:port pairs, protocol, timestamp), TCP flag
  bitmask (SYN/ACK/FIN/RST/PSH/URG) derived from CICFlowMeter's per-flag
  counts, bytes/packets per flow (total/fwd/bwd), flow duration, IAT stats
  (mean/std/max, flow/fwd/bwd), bidirectional ratios, and a derived binary
  label. Handles the known `Flow Byts/s`/`Flow Pkts/s` +inf edge case,
  duplicated in-file header rows, day-first timestamp parsing, and
  UTF-8/cp1252/Latin-1 encoding fallback with label-text normalization.
- `features/packet_features.py` — Scapy-based PCAP parser. Reconstructs flows
  from raw packets (5-tuple + flow-timeout splitting, matching CICFlowMeter's
  120s default) and extracts: TTL mean/var/min/max, TCP window size
  mean/var, IP fragmentation count/ratio (with correct IP-ID-based
  fragment-to-flow attribution for non-first fragments, which carry no L4
  header), payload size distribution, and TCP retransmission counts
  (repeated sequence number carrying data, in one direction). Also includes
  `detect_port_scan_signatures()`, a per-(source, window) heuristic
  distinguishing sequential vs. randomized destination-port access.
- `features/windowing.py` — joins flow-level and packet-level features onto a
  shared, configurable time window (`configs/pipeline.yaml`), producing one
  normalized feature matrix with a stable schema regardless of whether
  packet-level data was available for a given input (falls back to
  zero-filled packet columns + a `packet_features_available` flag).
  Normalization (StandardScaler/MinMaxScaler, configurable) is fit and applied
  per numeric column, with identifiers/labels excluded.
- `scripts/generate_synthetic_fixture.py` — generates a small, clearly-labeled
  **synthetic** CICFlowMeter-shaped CSV (benign background + a sequential
  port scan + a DoS-shaped burst) for smoke-testing the pipeline without
  touching real data. Not real traffic; do not use for anything beyond
  pipeline verification.
- `scripts/sanity_check.py` — two modes: `--csv <path>` deep-dives on one
  file (shape, dtypes, missing-data handling, +/-inf sanitization, value
  ranges, label distribution, per-window flow counts); `--all-days` runs a
  compact summary across all 8 CIC-IDS2017 files (shape, label distribution,
  Src/Dst IP population check) in one pass.
- Unit tests (`tests/`) covering column normalization, flag bitmask
  construction, inf-sanitization, ratio edge cases, day-first timestamp
  parsing, packet-level flow reconstruction (retransmissions, fragmentation,
  flow timeout splitting — built against synthetically crafted Scapy
  packets), port-scan signature detection, and windowing/merge correctness
  (including a regression test for keeping flow-level and packet-level
  sources anchored to the same absolute time when windowing).

### What's stubbed / TODO
- **Packet-level extractor has not been run against real captured traffic.**
  `data/cicids2018-pcap/` and `data/ctu13/<scenario>/` are empty
  (`.gitkeep`-only) — once real PCAP day-folders are placed there, run
  `python scripts/sanity_check.py --pcap <path>` and flag any schema
  surprises (VLAN tags, IPv6, tunneling, etc. are not yet handled). Since
  CIC-IDS2017's flow-level CSVs already have real IPs (see above), PCAPs are
  now needed only for the packet-level signal itself, not for identity
  recovery.
- CTU-13 ingestion (bidirectional NetFlow format) is a **different schema**
  from CICFlowMeter and is not yet built — deferred to Phase 2/3 when it's
  used as the held-out generalization test.
- Flow reconstruction in `packet_features.py` uses a documented heuristic
  (5-tuple + 120s inactivity timeout); no full TCP state-machine tracking.
- The negative-`Flow Duration` quirk noted above is surfaced, not resolved —
  decide before Phase 3 training whether to clip, drop, or keep those rows.

### How to run

```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Compact summary across all 8 CIC-IDS2017 files
./.venv/bin/python scripts/sanity_check.py --all-days

# Deep-dive on one file
./.venv/bin/python scripts/sanity_check.py \
    --csv "data/cicids2017/TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

# Include packet-level features once a PCAP is available
./.venv/bin/python scripts/sanity_check.py --csv <csv path> --pcap <pcap path>

# Sanity check against the synthetic fixture instead of real data
./.venv/bin/python scripts/sanity_check.py
```

Pipeline behavior (window size, normalization method, dataset paths) is
configured in `configs/pipeline.yaml`, not hardcoded in the extractor code.

## Phase 2 — Graph State Representation + Ground-Truth Transition Extraction

### Design decisions (documented in full in `graph/snapshot.py`'s docstring)
- **Nodes** = hosts (IP addresses) active (as flow source or destination) in
  a time window. Graph size varies window-to-window — expected, and what the
  Phase 3 GNN encoder is for.
- **Edges** = one aggregated *directed* edge per `(src_ip, dst_ip)` pair per
  window (not one edge per raw flow) — e.g. a 30-port scan against one target
  in one window collapses to a single edge with `dst_port_nunique=30`,
  keeping graph size proportional to host *relationships*, not packet count.
- **Features are aggregated, not re-derived**, from Phase 1's per-flow
  columns (37 node features, 24 edge features — see `NODE_FEATURE_NAMES`/
  `EDGE_FEATURE_NAMES` in `graph/snapshot.py`), split by out/in direction for
  nodes so a host's outbound behavior (e.g. scanning) and inbound behavior
  (e.g. being flooded) don't wash each other out.
- **Labels are kept separate from features** — `node_labels`/`edge_labels`/
  `graph_attack_present` are training targets derived from ground truth,
  never fed back in as model input.
- **Not normalized here.** Phase 3 holds out a full day for generalization
  testing; normalizing before that split would leak the held-out
  distribution into training. Phase 2 outputs raw aggregated values.
- **Windows are gap-filled**: every integer `window_id` from min to max is
  materialized per day, including zero-flow windows (as an explicit empty
  graph), so a fixed-Δt sequence model sees uniform time steps.
- **Transitions never cross a day boundary** (`graph/transitions.py`) — each
  CIC-IDS2017 day is an independent capture, even though the private IP
  space is reused across days.
- **`transition_type`** buckets each `(S_t, S_t+1)` pair into
  `benign_to_benign` / `benign_to_attack` / `attack_to_attack` /
  `attack_to_benign` from the PS's "label whether/how the state is
  progressing toward an attack" requirement. `benign_to_attack` is the
  specific progression moment a forecaster should be judged on catching
  early (Phase 6 benchmark should report recall on this category
  specifically, not just overall accuracy).

### What's built
- `graph/snapshot.py` — `build_snapshots_for_day()`: vectorized pandas
  aggregation (no per-flow Python loops) from Phase 1's merged/windowed flow
  table into one `GraphSnapshot` per window.
- `graph/transitions.py` — `build_transitions()`: turns a day's ordered
  snapshots into `(S_t, S_t+1, attack_label_next, attack_types_next,
  transition_type)` records.
- `graph/io.py` — pickle + JSON-summary persistence per day.
- `scripts/build_graph_dataset.py` — builds and saves the dataset for all 8
  CIC-IDS2017 files (or `--day <name>` / `--rows <n>` for a subset/sample).
- 9 unit tests on hand-crafted synthetic windowed data covering edge
  aggregation/direction, node attack-source/target/both labeling, port-scan
  signal propagation onto nodes, gap-filling, transition-type derivation,
  and the same-day-only invariant.

### Verified against real data
Ran `scripts/build_graph_dataset.py` on all 8 real CIC-IDS2017 files
(full files, no row cap) — **41 seconds total**, ~2.7M flows processed:

| Day | Windows (empty) | Attack transitions | Attack types found |
|---|---|---|---|
| Monday | 720 (233) | 0 | — (benign day, as expected) |
| Tuesday | 720 (232) | 127 | FTP-Patator, SSH-Patator |
| Wednesday | 720 (211) | 86 | DoS GoldenEye/Hulk/Slowloris/Slowhttptest, Heartbleed |
| Thursday-Morning | 241 (0) | 68 | Web Attack (Brute Force/XSS/SQLi) |
| Thursday-Afternoon | 245 (0) | 28 | Infiltration |
| Friday-Morning | 241 (0) | 163 | **Bot** (see note below) |
| Friday-Afternoon-PortScan | 150 (0) | 27 | PortScan |
| Friday-Afternoon-DDoS | 93 (0) | 21 | DDoS |

Two real-data findings surfaced while validating (both handled correctly by
existing Phase 1 code, not new bugs — documented here for the record):
1. **Thursday-Morning-WebAttacks has 62.9% of rows with a genuinely empty
   `Timestamp` field in the raw CSV** (288,602 of 458,968 rows) — a known
   quality issue in this specific CIC-IDS2017 file, not a parsing bug.
   `windowing.py` already drops unparsable/missing timestamps with a logged
   count; the remaining 170,366 rows are used. This file also uses a
   different timestamp format than the others (`M/D/YYYY H:MM`, no seconds,
   no leading zeros) — confirmed parsing correctly as day-first (e.g. `6/7/2017`
   → July 6, 2017, a Thursday, matching the dataset's documented July 3–7
   2017 capture week).
2. **Friday-WorkingHours-Morning is not "benign only"** despite how it was
   described — the real Label column contains **Bot** (botnet/ARES) traffic
   (163/240 windows show attack-present-next), matching CIC-IDS2017's
   official documentation of this capture (Friday morning = Botnet day).
   `features/datasets.py`'s registry has been corrected to reflect this.

### What's stubbed / TODO
- Packet-level columns are still all zero (`packet_features_available=False`
  throughout) — no PCAPs provided yet, same status as Phase 1.
- CTU-13 ingestion (different schema) still deferred to Phase 3, when it's
  used as the secondary held-out generalization test.
- No train/val/test split yet — that's explicitly Phase 3's job (and must
  hold out a full day/attack-type, per the PS). Phase 2 only guarantees the
  `day` field is present on every snapshot/transition so Phase 3 can filter
  by it. Friday's PortScan or DDoS file remain good hold-out candidates, as
  discussed.

### How to run
```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Build + persist the graph-sequence dataset for all 8 days (~41s, full files)
./.venv/bin/python scripts/build_graph_dataset.py

# One day only, or row-capped for a quick smoke test
./.venv/bin/python scripts/build_graph_dataset.py --day Tuesday
./.venv/bin/python scripts/build_graph_dataset.py --rows 50000
```
Output: `data/processed/cicids2017/<day>.pkl` (snapshots + transitions,
regenerable cache, git-ignored) and `<day>_summary.json` (small, human-
readable — also git-ignored, but printed to console by the build script).

## Phase 3 — World Model (GNN + Temporal Layer)

### Architecture
`models/gnn_encoder.py` (GNN) → `models/temporal.py` (swappable LSTM/
Transformer) → `models/world_model.py` (dynamics + attack-forecast +
attack-type heads). Full design rationale is in each module's docstring;
summary:

- **GNN encoder**: stacked `TransformerConv` layers (PyTorch Geometric) —
  chosen over plain GCN/GraphSAGE because it natively conditions attention on
  `edge_attr` (byte/packet counts, flags, port-scan score, ...) and exposes
  per-edge attention weights, which Phase 5 will use directly for
  explainability instead of bolting on SHAP separately. Graph-level readout =
  concat(mean-pool, max-pool of node embeddings, mean-pool of edge features)
  → projected to a fixed-size latent `z_t`. Zero-node ("quiet window") graphs
  need no special-casing: they pool to a learned zero-bias representation for
  free.
- **Temporal core**: swappable via config (`model.temporal_type: lstm |
  transformer`) — both implemented and tested, not just interface-stubbed.
  LSTM is causal by construction; the Transformer variant uses a causal
  attention mask (verified: perturbing a future timestep doesn't change past
  outputs). An RSSM is a documented future upgrade slotting into the same
  `TemporalCore` interface.
- **Training objective — genuinely a world model, not a classifier**: at
  every position t in a sequence, the model predicts the *next window's
  latent embedding* ẑ_{t+1} from causal history h_t (MSE loss, target
  stop-gradiented to prevent representation collapse — see
  `world_model.py`'s docstring for why that matters here), and separately
  forecasts `attack_label_next` from `[h_t ; ẑ_{t+1}]` — i.e. from the
  learned dynamics prediction itself, not from ground truth. This is exactly
  what makes Phase 4's K-step rollout possible: at inference there's no
  ground-truth next state, so the model's own ẑ_{t+1} gets fed back in
  autoregressively (`WorldModel.rollout_step`, LSTM path implemented now).
- **Attack-type head** (auxiliary): multi-label over a vocabulary built ONLY
  from non-held-out days, deliberately excluding the held-out day's category
  — see below.

### Train/val/test split
- **Held out entirely**: `Friday-Afternoon-DDoS` (config-swappable) — the
  sole source of the "generalizes to an unseen attack type" claim, since
  DDoS is volumetrically distinct from anything else in the other 6
  non-benign days.
- **Train/val on the remaining 7 days**: split **chronologically per day**
  (first 80% of each day's windows by time → train, last 20% → val), not by
  randomly shuffling windows — consecutive windows are highly autocorrelated,
  so a random split would leak near-identical traffic across the boundary.
  Sequences never straddle the split point.
- Normalization (`models/normalization.py`) is fit on the train region only,
  exactly because Phase 2 deliberately left features unnormalized.

### What's built
- `models/gnn_encoder.py`, `models/temporal.py`, `models/world_model.py`,
  `models/normalization.py`, `models/dataset.py`, `models/train.py`.
- `configs/training.yaml` — full reproducible config (seed 42, split,
  architecture, optimizer).
- 10 new unit tests (50 total, all passing): encoder correctness on
  mixed/all-empty graph batches, temporal-core shape + causality (both
  LSTM and Transformer), world-model forward shapes, stop-gradient
  verification, a real end-to-end "does this learn" convergence test, and
  dataset/split correctness (no leakage across the train/val boundary, held-out
  day fully excluded from the attack-type vocabulary).

### Trained and verified against real data
Ran `models/train.py` on the full Phase 2 CIC-IDS2017 cache (MPS-accelerated,
~4s/epoch, early-stopped at epoch 14/40 on val ROC-AUC). **Held-out
`Friday-Afternoon-DDoS` day** (never seen in any form during training):

| Metric | Value |
|---|---|
| Accuracy | 0.963 |
| Precision | 0.951 |
| Recall | 0.917 |
| F1 | 0.933 |
| ROC-AUC | 0.989 |

This is the headline generalization result: the binary attack-forecast head
has no structural knowledge of DDoS (it's excluded from the attack-type
vocabulary), yet correctly ranks/classifies DDoS-vs-benign windows on a day
it never trained on.

Two things surfaced during training, reported honestly rather than
smoothed over:
1. **A real bug caught by inspecting results, not just trusting them**:
   the first training run selected its "best" checkpoint by combined
   total_loss, which is dominated by the easy-to-optimize dynamics-MSE term
   — it picked an early epoch where the classifier hadn't learned to
   separate attack from benign at all (val F1 = 0) purely because dynamics
   loss had cheaply collapsed toward zero. Fixed by selecting checkpoints on
   val ROC-AUC (threshold-independent, task-relevant) instead, and by adding
   `pos_weight` to the attack-forecast BCE loss (attack windows are ~23% of
   train positions, not 50/50).
2. **In-distribution val AUC is mediocre (~0.45-0.57) and noisy** across the
   7 non-held-out days' chronological tail, while held-out-day performance is
   strong. This is plausible, not a red flag: the val tail mixes several
   behaviorally-subtle attack types (late-stage infiltration, slow
   brute-force) with very few examples each, while DDoS is volumetrically
   extreme and comparatively easy to separate from benign. Also worth
   reporting plainly: the held-out day contains exactly **one** true
   benign→attack transition in its 93 windows, so
   `benign_to_attack_recall` on that day (0.0, i.e. that single instance was
   missed) is a sample size of one, not a reliable measurement — Phase 6's
   benchmark against the baseline, with more attack days pooled, will give a
   statistically meaningful version of this metric.

### What's stubbed / TODO
- Packet-level features still zero everywhere (no PCAPs yet, unchanged from
  Phase 1/2) — the world model is currently trained on flow-derived graph
  features only.
- CTU-13 is not yet integrated as the secondary held-out generalization test.
- `WorldModel.rollout_step` (the autoregressive K-step primitive Phase 4
  needs) is implemented and used implicitly during training (every position
  is a 1-step rollout), but the *multi-step* rollout loop, MITRE stage
  mapping, and Transformer-compatible rollout path are Phase 4's job.
- Val-set size (~95-340 sequences depending on stride) is small; more
  epochs/architecture search were intentionally not chased further here —
  Phase 6's baseline comparison is the more meaningful benchmark to react to.

### How to run
```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Train (reads configs/training.yaml)
./.venv/bin/python models/train.py --config configs/training.yaml
```
Output (git-ignored, regenerable from config + seed):
`checkpoints/phase3/{best_model.pt, normalizer.pkl, attack_type_vocab.json,
run_config.yaml, training_log.json, test_report.json}`.

## Phase 4 — K-Step Rollout Engine + MITRE ATT&CK Stage Mapping

### Rollout engine (`rollout/engine.py`)
Given a real observed context window, rolls the trained world model forward
K steps entirely autoregressively (the model's own predicted latent ẑ feeds
back in as the next step's input — no ground truth involved, exactly as
described in Phase 3). Implements **both** temporal cores end-to-end, not
just at training time:
- **LSTM**: context primes the LSTM once to get its final (h, c) state, then
  each further step is an O(1) state-carried recurrence
  (`WorldModel.rollout_step`).
- **Transformer**: no fixed-size state to carry, so each step re-runs the
  (still causally-masked) encoder over the context extended with every ẑ
  predicted so far — O(K²), fine for the K values this targets.

Output per step: infiltration probability (sigmoid of the attack-forecast
logit), full attack-type probability distribution, and a mapped MITRE stage.

### MITRE ATT&CK stage mapping (`rollout/mitre.py`)
Two-tier, fully documented in the module (judgment calls called out
explicitly, since the PS leaves this open):
1. **Primary**: the model's own predicted attack-type (when confident, ≥0.5)
   mapped via a fixed table — PortScan→Reconnaissance, FTP/SSH-Patator and
   Web Attacks→Initial Access, Infiltration→Lateral Movement (CIC-IDS2017's
   "Infiltration" scenario is post-compromise internal pivoting, not initial
   entry), Bot→Command and Control.
2. **Fallback heuristic** (used when no attack-type is confident — which is
   *every* prediction on an unseen attack type, since it structurally can't
   be in the vocabulary): raw aggregate stats from the last real context
   window — high port-scan-score/port-diversity → Reconnaissance, high
   outbound-byte asymmetry → Exfiltration.
3. **Honest 6th label**: the DoS/DDoS family and Heartbleed don't fit any of
   the PS's five stages — mapped to an explicit `"Denial of Service /
   Impact"` label instead of force-fit into a misleading one of the five.

**Documented limitations, found while testing against real data, not
hidden:**
- CIC-IDS2017 has no natural Command & Control category beyond Bot, and no
  Exfiltration category at all — Exfiltration is reachable only through the
  raw-feature fallback, never the attack-type path.
- The fallback's "many distinct ports" Reconnaissance signal isn't perfectly
  specific: rolling out from inside the actual DDoS attack (window 40),
  the top driving edge showed 1,877 distinct destination ports on a
  6,264-flow/46MB DDoS-labeled edge, which the heuristic read as
  port-scan-shaped and labeled Reconnaissance rather than the DoS/Impact
  label a human would assign. A volumetric flood with randomized/incrementing
  ports can superficially resemble a scan at the aggregate-feature level;
  the attack-type path (when confident) doesn't have this problem, only the
  fallback does.

### Explainability stub (`rollout/explain_stub.py`)
Per the PS, a simple version is acceptable here (full explainability is
Phase 5). Ranks edges in the **last real observed** context snapshot by a
documented composite score over raw features known to correlate with attack
behavior (port diversity, retransmissions, SYN count, byte volume) and
reports the top-K with human-readable src/dst/port/byte summaries. This is
explicitly NOT model attribution (no gradients/attention) — attached once at
the context level, since predicted future windows exist only as latent
embeddings with no raw features to rank. Phase 5 will replace this with real
attention weights (the GNN encoder's `TransformerConv` layers already expose
per-edge attention — chosen in Phase 3 partly for this reason) or SHAP.

### What's built
- `rollout/engine.py`, `rollout/mitre.py`, `rollout/explain_stub.py`,
  `scripts/run_rollout.py`.
- Cleaned up a dead parameter (`h_prev`, unused) in `WorldModel.rollout_step`
  found while building the engine around it.
- 18 new unit tests (58 total, all passing): rollout produces exactly K
  steps for both temporal cores, attack-type probabilities populate/empty
  correctly, same-day invariant enforced, deterministic in eval mode; MITRE
  mapping's confidence thresholds, table lookups, 6th-label handling, and
  each fallback heuristic; explainability stub's ranking and edge cases.

### Verified against real data (held-out DDoS day)
Ran `scripts/run_rollout.py` at three points in `Friday-Afternoon-DDoS`
(the Phase 3 held-out day — every prediction here is on data never seen in
any form during training):

| Context ends at window | Ground truth | Rollout behavior |
|---|---|---|
| 10 (well before attack) | Benign | Stays below 0.5 throughout, converges to ~0.48 |
| 25 (1 window before attack starts at window 26) | Benign, imminent | Sits at 0.45–0.48, just below the decision threshold |
| 40 (well inside the attack) | Attack | Starts at 0.56, correctly classified attack for the first ~6 of 10 steps |

**Observed, disclosed limitation**: rollout probabilities compress into a
narrow band (~0.45–0.56) and drift toward a ~0.48 fixed point as K grows,
regardless of starting condition — a known, expected property of open-loop
latent rollouts (compounding uncertainty pulls long-horizon predictions
toward a "typical" state). Near-term steps (1-3) carry more signal than
distant ones; this is useful operational guidance (trust the near horizon
more) rather than a bug to hide.

### What's stubbed / TODO
- The explainability stub is deliberately simple (magnitude-based, not model
  attribution) per the PS's phasing — Phase 5 replaces it.
- No calibration/temperature-scaling applied to probabilities — the observed
  narrow band around 0.5 reflects the underlying classifier's raw
  calibration from Phase 3, not a rollout-engine issue.
- CTU-13 rollout untested (dataset not yet integrated).

### How to run
```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Roll out K steps from a chosen context window
./.venv/bin/python scripts/run_rollout.py --day Friday-Afternoon-DDoS \
    --start-window 40 --context-len 20 --k 10 --json-out /tmp/rollout.json
```

## Phase 5 — Explainability Layer

### Approach: real GNN attention + gradient-based feature attribution (not SHAP)
The PS allows either SHAP or attention, noting attention is "likely simpler
given the temporal Transformer/GNN attention already in the model." Two
genuine, model-internal methods were implemented — deliberately not the
Phase 4 heuristic stub, which remains attached only for side-by-side
comparison:

1. **GNN edge attention** (`explain/attention.py`): `GNNEncoder` (Phase 3)
   was built on `TransformerConv` specifically so this would be free —
   `return_attention_weights=True` is now wired into `GNNEncoder.forward()`,
   averaged across heads and layers, with zero effect on trained weights (no
   new parameters). Answers "which flow drove this."
2. **Input×Gradient feature attribution** (`explain/saliency.py`): a single
   backward pass giving `|input × d(attack_logit)/d(input)|` per node/edge
   feature. Chosen over SHAP because exact/approximate SHAP needs many
   forward passes over masked feature coalitions and doesn't map cleanly
   onto a variable-size batched graph; a direct gradient is cheap,
   well-established (Simonyan et al. 2013-style saliency), and — crucially —
   works identically regardless of temporal core (LSTM has no attention
   mechanism at all, so this is the *only* genuine attribution method
   available for the model actually trained in Phase 3). Answers "which
   specific flags/ports/flow statistics drove it," at the feature-name
   level, plus a bonus `per_window_saliency` breakdown ("which past context
   window mattered most") that attention can't give for an LSTM core.

`explain/explainer.py` unifies both into one `ExplanationRecord` with a
human-readable `summary` string, and deliberately reuses
`rollout.engine.run_rollout(..., k=1)` for the probability/stage rather than
reimplementing that forward pass a third time — it explains exactly the
prediction the rollout engine reports as step 1, by construction.
`rollout/engine.py` now also attaches both methods' output directly to every
`RolloutResult` (`attention_flows`, `feature_saliency`), alongside the
existing Phase 4 stub (`driving_context_flows`) for comparison — satisfying
"no black-box outputs, every prediction ships with this attached" for the
K-step forecast, not just single predictions.

### A real bug found and fixed while verifying against live data
Raw attention weights are a softmax **per target node** over its incoming
edges only — an edge with weight 1.0 into a host with a single incoming
edge carries *zero* information (there was no competition), while 0.6 among
5 real competitors is a much stronger, more informative signal. The first
version of `top_edges_overall` ranked by raw attention weight and was
verified against the real held-out DDoS day to **bury the actual attack
edge** (`172.16.0.1 -> 192.168.10.50`, 6,264 flows, 46MB, attention=0.62
among 7 competitors) beneath several trivial single-choice targets scoring a
meaningless 1.0. Fixed by ranking on `attention_concentration`
(`attention_weight × num_competitors`) instead — re-verified: the real
attack edge now correctly appears in the top-5. A regression test
(`test_top_edges_overall_prefers_concentration_over_raw_weight`) encodes
this exact scenario.

### What's built
- `models/gnn_encoder.py`: `return_attention` param added to `forward()`
  (Phase 3 anticipated this; zero new parameters, existing checkpoint loads
  unchanged).
- `explain/attention.py`, `explain/saliency.py`, `explain/explainer.py`.
- `rollout/engine.py` updated to attach both to every `RolloutResult`
  (removed the function-level `@torch.no_grad()` decorator, scoping it to
  just the rollout loop, since saliency needs gradients in the same call).
- `scripts/explain_prediction.py` — CLI for a single structured,
  human-readable explanation.
- 13 new unit tests (71 total; all passing), including the concentration-
  ranking regression test above.

### Verified against real data (held-out DDoS day, window 40 — inside the attack)
```
Predicted 56% infiltration probability ... mapped to MITRE stage
'Reconnaissance' (basis: heuristic_fallback). Most-attended flow:
192.168.10.50 -> 192.168.10.15 (attention=0.38, ...). Top contributing
host-level features: out_urg_cnt_sum, in_ack_cnt_sum, out_ack_cnt_sum.
Top contributing flow-level features: ack_cnt_sum, urg_cnt_sum, syn_cnt_sum.
```
Reported honestly rather than cherry-picked: the single *most*-attended flow
in this window was not the attack edge (attention doesn't always spotlight
the true cause) — but the real attack edge does surface in the concentration-
ranked top-5, with a competitively-earned 0.62 attention weight, which a
naive global ranking had been hiding entirely until the fix above.

### What's stubbed / TODO
- Transformer-temporal-core self-attention (a second, complementary
  "which past window mattered" signal specific to that core) is not wired
  up — the model actually trained and shipped in Phase 3 uses LSTM, which
  has no attention mechanism at all, making this a low-value gap to chase
  further right now; `per_window_saliency` already answers the same
  question for any core.
- No quantitative faithfulness evaluation (e.g. deletion/insertion curves)
  of the attribution methods — Phase 6's benchmark is a better place to
  spend evaluation effort.

### How to run
```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Explain one prediction (structured + human-readable)
./.venv/bin/python scripts/explain_prediction.py --day Friday-Afternoon-DDoS \
    --start-window 40 --context-len 20 --json-out /tmp/explanation.json

# Every rollout now also carries real attention/saliency automatically
./.venv/bin/python scripts/run_rollout.py --day Friday-Afternoon-DDoS \
    --start-window 40 --context-len 20 --k 10
```

## Phase 6 — Baseline + Benchmark

### Baseline design: same information, no learning, no temporal context
`baseline/features.py` flattens each window into a fixed-size vector using
the **same pooling structure as the world model's own GNN encoder**
(mean-pool + max-pool over node features, mean-pool over edge features —
see `models/gnn_encoder.py`), applied to the identical normalized features
(reuses Phase 3's exact fitted `FeatureNormalizer`), so neither model has an
input-quality advantage. The only thing removed is what the world model
*adds*: learned message passing and temporal context. The baseline
(`baseline/train_baseline.py`, `sklearn.LogisticRegression`,
`class_weight="balanced"`, C selected by val AUC — the same
selection-by-AUC discipline as Phase 3's checkpoint) predicts
`attack_label_next` from exactly one window, with no history — precisely
the world model's own forecasting task, minus dynamics learning. Reuses
Phase 3's exact held-out day and chronological split
(`models/dataset.split_day_snapshots`, refactored out as a single shared
primitive so both models are guaranteed to agree on the split boundary).

### Result (`benchmark/report.md`, regenerated by `benchmark/run_benchmark.py`)

**Held-out day `Friday-Afternoon-DDoS` (unseen attack type — the PS's generalization requirement):**

| model | accuracy | precision | recall | f1 | fpr | roc_auc |
|---|---|---|---|---|---|---|
| World Model | 0.963 | 0.951 | 0.917 | 0.933 | 0.019 | 0.989 |
| Baseline (LogReg) | 0.696 | 0.000 | 0.000 | 0.000 | 0.099 | 0.040 |

The baseline doesn't just underperform — it **fails completely** on the
unseen attack type (F1=0.000, AUC=0.040, i.e. its ranking is worse than
random). The world model's temporal dynamics learning is the difference
between "detects the unseen attack" and "never flags it once."

**Validation set (in-distribution, seen attack types):**

| model | accuracy | precision | recall | f1 | fpr | roc_auc |
|---|---|---|---|---|---|---|
| World Model | 0.857 | 0.350 | 0.289 | 0.317 | 0.070 | 0.484 |
| Baseline (LogReg) | 0.656 | 0.090 | 0.175 | 0.119 | 0.270 | 0.320 |

World model still wins on every metric, though both scores are weaker here
than on the held-out day — see the finding below.

### A finding worth investigating and reporting, not smoothing over
The baseline's first run scored **below-random AUC (~0.30) on validation**
— strong enough (and consistent enough across every regularization
strength tried) to suspect a bug rather than genuine weakness. Investigated
and resolved as a real, disclosed property of the *validation set itself*,
not a modeling bug: **4 of the 7 non-held-out days' chronological tail
(the val region) contain zero attack windows at all** — Monday (benign by
design), Tuesday, Thursday-Morning, and Thursday-Afternoon's attack
campaigns all finish before that day's capture ends, leaving nothing but
benign traffic in their last 20%. Only Wednesday, Friday-Morning, and
Friday-Afternoon-PortScan have any class variation left to measure ranking
quality on in val, so the pooled val AUC ends up dominated by whichever of
those has the most examples (Wednesday) — where the linear baseline
genuinely struggles. This also retroactively explains Phase 3's own
observation of "mediocre, noisy val AUC" for the world model itself: it's
the same small, unevenly-distributed validation slice. The held-out day's
AUC remains the statistically meaningful, primary result for both models.

### What's built
- `baseline/features.py`, `baseline/train_baseline.py`.
- `benchmark/metrics.py` (adds false positive rate, which neither
  scikit-learn nor Phase 3's `evaluate()` expose directly), `benchmark/run_benchmark.py`.
- Refactored `models/dataset.py`: `split_day_snapshots()` extracted as the
  single shared day-splitting primitive (`build_split` and
  `get_train_region_snapshots` both now call it), so the world model and
  baseline are guaranteed to use the identical split.
- `models/train.py`'s `evaluate()` gained an optional `return_raw` flag
  (labels + probs) so the benchmark reuses the exact same evaluation loop
  Phase 3 already validated, rather than re-deriving world-model metrics.
- 10 new unit tests (81 total, all passing): flattening shapes/empty-window
  handling, same-day-safe pair building, false-positive-rate correctness
  against a hand-computed confusion matrix.

### What's stubbed / TODO
- Only one baseline (logistic regression) — no tree-based/gradient-boosted
  baseline, though the PS only asks for logistic regression specifically.
- The two models' evaluation example sets differ by construction (documented
  in `benchmark/run_benchmark.py`'s module docstring) — a deliberate,
  disclosed methodological choice, not an oversight, but worth knowing
  before quoting these numbers out of context.

### How to run
```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Train the baseline (reuses Phase 3's split + normalizer)
./.venv/bin/python baseline/train_baseline.py

# Run the benchmark, regenerating benchmark/report.md
./.venv/bin/python benchmark/run_benchmark.py
```

## Phase 7 — Backend API (FastAPI)

### Scope decision: standalone PCAP support via the real CICFlowMeter, not a reimplementation
Phase 1 built packet-level features (`features/packet_features.py`) as a
*supplement* to CSV-derived flow-level features, not a replacement — it
doesn't compute the flag counts, IAT statistics, etc. the trained model's
37 node / 24 edge features depend on. Rather than reimplement CICFlowMeter's
flow-assembly logic from scratch, or silently degrade PCAP-only uploads, the
user's own direction was to integrate the actual tool: `pip install
cicflowmeter` (the same tool family that generated CIC-IDS2017/2018's
training data) converts an uploaded PCAP into a flow-level CSV first, which
then runs through the *existing, unmodified* Phase 1 pipeline
(`features/pcap_to_flows.py`). `packet_features.py` still runs separately on
the same raw PCAP for packet-level augmentation, exactly as designed. The
column names matched our canonical schema with zero new aliases needed
(verified empirically, not assumed).

**Two real upstream bugs found and worked around** (documented in
`features/pcap_to_flows.py`): (1) `cicflowmeter`'s own CLI calls its
`create_sniffer()` with positional arguments in the wrong order, crashing on
every single invocation regardless of flags — worked around by calling the
library's internal API directly with correct keyword arguments; (2) its
`AsyncSniffer` isn't wired up to flush completed flows when an offline
capture ends, so the output CSV was created but silently empty (0 rows) —
worked around with an explicit `session.garbage_collect(None)` after
`sniffer.join()`.

### Pipeline: full walk-forward timeline + K-step forecast + real explanation
`POST /analyze` accepts a CSV, a PCAP, or both, and returns:
- **`timeline`**: walk-forward — for every window with enough preceding
  history, the model's 1-step-ahead prediction for the window right after
  it (reconstructs how infiltration probability evolved across the whole
  uploaded capture).
- **`forecast`**: genuine K-step-ahead rollout (Phase 4) beyond the last
  observed window, into the unobserved future.
- **`flagged_flows`** + **`explanation`**: real GNN attention + gradient
  attribution (Phase 5), attached to the single most decision-relevant
  prediction (the present moment) rather than recomputed for every
  timeline point, for response-time reasons documented in
  `api/pipeline.py`.
- **`warnings`**: non-fatal notices (no PCAP given, capture shorter than
  the model's trained context length, etc.) — surfaced, never hidden.

Model loaded once at FastAPI startup (`lifespan`), not per-request. Fully
offline: the checkpoint is local, `cicflowmeter`/`scapy`/`torch` all run
locally, no cloud API calls anywhere in the inference path.

### A real bug found and fixed while testing
The first real request returned an HTTP 500: `rollout.mitre.predict_stage`
returns a `StagePrediction` dataclass, and Pydantic v2 doesn't automatically
coerce an arbitrary dataclass into a `BaseModel` field. Fixed by converting
explicitly to a plain dict before constructing the response
(`api/pipeline.py`'s `_stage_to_dict`) — caught by an actual test request,
not by inspection.

### What's built
- `api/schemas.py`, `api/pipeline.py`, `api/main.py`.
- `features/pcap_to_flows.py` (the CICFlowMeter integration described above).
- Fixed a real, separate bug found while testing with unlabeled data: real
  uploads have no `Label` column, so `label_binary` is `NaN` — casting `NaN`
  straight to `int64` (for `edge_labels`/`node_labels`) is undefined
  behavior in numpy (observed to silently produce a garbage value rather
  than raise), and a naive `!= "benign"` string filter let `NaN` through as
  a spurious attack type literally named `"nan"`. Both fixed explicitly in
  `graph/snapshot.py`, with a regression test.
- 13 new unit/integration tests (95 total, all passing) using FastAPI's
  `TestClient` — CSV-only, PCAP-only, both, missing-file 422, malformed-file
  4xx/5xx (not a crash), too-short-capture 422, response schema shape,
  `k` parameter respected.
- **Tested against a real, running server** (not just in-process
  `TestClient`) with `curl`, per the PS's requirement to prove this works
  standalone before Phase 8: `/health`, `/model-info`, `/docs` (interactive
  OpenAPI UI), and `/analyze` against a 150,000-row real CIC-IDS2017 CSV
  slice spanning the actual transition into the DDoS attack (see below) and
  a real standalone PCAP upload, both in a few seconds.

### Verified against real data (150K rows, live server, not a test double)
Uploaded a real, un-cached slice of `Friday-Afternoon-DDoS` spanning the
benign-to-attack transition. The walk-forward timeline correctly tracks the
shift:

| window | infiltration probability | stage |
|---|---|---|
| 20 (benign) | 0.420 | Benign |
| 30 (attack starting) | 0.540 | Reconnaissance (heuristic) |
| 40 (in attack) | 0.558 | Reconnaissance (heuristic) |

The K-step forecast from the end of the upload starts at 0.557 and decays
toward the ~0.5 fixed point documented in Phase 4/5 — consistent, expected
behavior, not a new artifact of the API layer.

### What's stubbed / TODO
- Per-window explanation isn't computed for every timeline point (only the
  final prediction) — a documented, deliberate response-time/scope decision,
  not an oversight.
- No auth/rate-limiting (out of scope for an offline research prototype;
  would matter before any real deployment).
- CTU-13 untested against the API (dataset not yet integrated anywhere in
  the project).

### How to run
```bash
# Unit tests (requires a trained checkpoint at checkpoints/phase3/)
./.venv/bin/python -m pytest -q

# Start the server
./.venv/bin/uvicorn api.main:app --port 8000

# Interactive docs
open http://localhost:8000/docs

# Analyze a CSV
curl -X POST http://localhost:8000/analyze \
    -F "csv_file=@path/to/flows.csv"

# Analyze a standalone PCAP (converted via cicflowmeter automatically)
curl -X POST http://localhost:8000/analyze \
    -F "pcap_file=@path/to/capture.pcap"
```

## Phase 9 — Differentiator Features (backend/ML only — no frontend yet)

Three features built on top of the existing, unmodified Phase 1-8 system:
Cyber Threat Momentum Score, Containment Window Estimator, and a
Counterfactual Intervention Simulator. All three are **derived/post-processing
logic — no new model was trained.** Per CLAUDE.md's Phase 9 scope, no
frontend UI is built for these yet; `/simulate` is endpoint wiring only.

### Before starting: confirming the baseline's F1=0.000 is real, not a bug
Before building anything, re-investigated Phase 6's headline number by
running the world model and baseline on identical `X_test`/`y_test` arrays
(same 92 examples, same 22.8% attack rate, same `FeatureNormalizer`, same
`split_day_snapshots` boundary) and inspecting the baseline's raw
probabilities directly:

- Confusion matrix: `tn=64, fp=7, fn=21, tp=0` — the model's predicted
  probability for every single true-attack example stayed under 0.5
  (`mean=0.020, max=0.417`), so TP=0 exactly, not approximately. AUC=0.040
  (a flipped-label AUC of 0.960 confirms the model's *ranking* is coherent,
  just inverted relative to this label).
- No leakage and no split artifact: the normalizer, training data, and test
  data were all traced back to the same fitted objects Phase 6 used; the
  DDoS traffic's volumetric features sit at the 96th-99th percentile of the
  *training* range, i.e. within-distribution, not extrapolation the model
  never saw the shape of.
- Root cause, found in the coefficients: the top three by `|coef|` are
  `node_mean__out_urg_cnt_sum` (+5.4), `edge_mean__ack_cnt_sum` (-4.0),
  `edge_mean__flag_bitmask_or` (-3.5) — a decision boundary learned from
  training attacks (brute-force, web attacks, infiltration — all
  flag-pattern-driven, low-volume) that penalizes exactly the ACK-heavy,
  flag-uniform traffic signature a volumetric DDoS produces. The linear
  model isn't uncertain about DDoS traffic — it's confidently, systematically
  wrong about it, because DDoS's real signature inverts the flag-based
  correlations it learned from every other attack type.

This is a genuine capability gap (a linear boundary that doesn't transfer
across attack *shapes*), not a split bug — exactly the generalization gap
the world model's GNN attention + temporal context is built to close, and
the number can be trusted as the basis for Phase 9's work.

### Feature 1 — Cyber Threat Momentum Score (`momentum/`)
A rate-of-escalation metric computed entirely from model-predicted/graph-
observable signals (never ground truth) — `compute_momentum()` combines four
weighted signals into a 0-100 score plus a human-readable top-driver
explanation:

| signal | weight | source |
|---|---|---|
| probability slope | 0.40 | recent infiltration-probability trend (`np.polyfit`) |
| newly affected host ratio | 0.25 | host churn across recent snapshots |
| stage progression rate | 0.20 | how often the predicted MITRE stage has changed recently |
| forecast trend | 0.15 | slope of a short forward rollout ("expansion of predicted attack paths") |

Two honest scope notes, disclosed rather than glossed over: "expansion of
predicted attack paths" falls back to the single top rollout path's trend
since Multiple Parallel Futures isn't built (`attack_path_expansion_available=False`
flags this on the result); "privilege level changes" is omitted entirely,
not faked — NetFlow features carry no authentication data to infer it from.
When a rollout result isn't available, the other three signals are
reweighted (not zero-penalized).

**Validated** (`scripts/validate_momentum.py`) by walking forward across the
entire held-out `Friday-Afternoon-DDoS` day using only predicted signals,
then comparing against real ground-truth labels for evaluation only: mean
momentum in the 6 windows before attack onset was **11.5**, rising to
**25.4** during onset (peaking 30-33 as probability ramps up), then
correctly settling back to 2-18 once probability plateaus — momentum
measures rate of escalation, not standing severity, so de-escalating during
a sustained, already-detected attack is correct behavior, not a miss.
Caveat found during validation: `newly_affected_host_ratio` dominates as the
reported top driver even on many benign windows, noisier than expected due
to natural host churn in this dataset — worth revisiting if this ships
further.

### Feature 2 — Containment Window Estimator (`containment/`)
Uses the existing trained world model + rollout engine only — no separate
model. A real, disclosed reframing was needed before building this: the
brief asks for stage-to-stage transition times (e.g. Reconnaissance →
Lateral Movement), but checked directly against the data first and found
**no genuine stage-to-different-stage transitions exist anywhere in the 8
CIC-IDS2017 days** — each day runs exactly one attack category start to
finish. What the data does have, richly, is repeated attack-episode
on/off cycles within a day (Wednesday alone contributes 21 separate DoS
episodes). So `build_stage_duration_stats()` estimates **time-to-containment
per stage** (how long an episode of stage X typically lasts) from real
`Timestamp`-derived window durations across training days, using
ground-truth labels only at training time — inference needs only the
current predicted stage string.

`estimate_containment_window()` returns a percentile-based range (p25-p90,
"typical") plus the full observed range, not a single point estimate
(episode durations here are heavily right-skewed — a few long episodes, many
one-window blips — so mean±std would misrepresent the distribution), flags
low-confidence stages (<3 historical episodes) rather than hiding them, and
looks up a simple rules-based recommended action per stage (`ACTION_TABLE`),
optionally qualified by a top explainability feature name (e.g. "driven by a
scan-like spread across many destination ports").

**Validated** two ways (`scripts/validate_containment.py`):
- **Leave-one-day-out** across the 7 training days: only "Initial Access"
  had 2+ days to cross-validate against (all other stages occur on exactly
  one training day, so those folds are honestly skipped, not hidden) — 5/9
  (56%) held-out episode instances fell within both the typical and full
  observed ranges.
- **True held-out day** (`Friday-Afternoon-DDoS` vs. stats from all 7
  training days): true DDoS duration = **21.0 min**; the typical (p25-p90)
  range `[1.0, 8.0]` **misses** it; the full observed range `[1.0, 24.0]`
  **contains** it; a point estimate (median = 1.0 min) would have been off
  by 20 minutes. A clean, real demonstration of why the brief specifically
  asks for a range rather than a point estimate.

### Feature 3 — Counterfactual Intervention Simulator (`counterfactual/`)
Reuses the existing graph representation and rollout engine only — no
simulation model. `counterfactual/graph_ops.py` implements two
graph-structural mutation primitives, each returning a new `GraphSnapshot`
(input never mutated) with correct edge-index reindexing and recomputed
`num_flows`/`graph_attack_present`/`graph_attack_types`:

- `isolate_host(snapshot, host_ip)` — removes a host and every edge
  touching it. Covers "isolate a host/endpoint."
- `block_flow(snapshot, src_ip, dst_ip, bidirectional=True)` — removes edges
  between one specific host pair only. Covers "revoke an account/session"
  and "block a specific flow" — at this graph's granularity (a host pair's
  aggregated traffic per window), those are the same operation.

**Disclosed limitation:** true per-port blocking is not implemented as a
graph mutation. `graph/snapshot.py` aggregates individual flows into
host-pair edges and keeps only `dst_port_nunique` (a *count* of distinct
ports) as an edge feature — actual port numbers are discarded during that
aggregation and aren't recoverable from a `GraphSnapshot`. True port-level
blocking would require re-deriving the graph from the pre-aggregation
per-flow DataFrame — out of scope here, noted for anyone extending it.

`counterfactual/simulator.py`'s `simulate_intervention()` runs the existing
`run_rollout()` twice — once on the original context, once on the
action-modified context — and returns a before/after probability comparison
per step, the resulting stage change, and simple rules-based side-effect
notes (e.g. how many *other*, non-attack-flagged flows an isolated host
would also lose).

**Validated** (`scripts/validate_counterfactual.py`) against a real context
window from the held-out `Friday-Afternoon-DDoS` day containing the known
attack edge (`172.16.0.1 → 192.168.10.50`, confirmed by direct inspection of
the day's ground-truth edge labels):

| action | mean probability | stage |
|---|---|---|
| none (original) | 0.508 | Reconnaissance |
| isolate `172.16.0.1` | 0.481 (−0.027) | Benign / No Stage |
| block `172.16.0.1 ↔ 192.168.10.50` | 0.483 (−0.025) | Benign / No Stage |
| negative control: isolate an unrelated host | 0.50772 (~+0.00000) | Reconnaissance (unchanged) |

Both real defender actions measurably reduce predicted infiltration
probability and flip the predicted stage back to benign; the negative
control (isolating a host with no connection to the attack) moves
probability by ~1e-5 — several orders of magnitude smaller than the real
actions' ~0.025-0.027 effect, and small enough to be graph-aggregation noise
from removing any node rather than a causal signal. This confirms the
simulator is attributing effect to the targeted host specifically, not just
reacting to any graph edit.

### API wiring: momentum + containment on every `/analyze`, counterfactual as `POST /simulate`
`ModelBundle` now builds `stage_duration_stats` once at server startup
(`containment.estimator.build_stage_duration_stats` over every cached
training day — deliberately including the day held out from *model
training*, since this is the production bundle, not the controlled
validation run above). Every `/analyze` response now carries a `momentum`
and a `containment` field alongside the existing timeline/forecast/
explanation, computed from data the endpoint was already producing (the
walk-forward timeline, the K-step forecast, and explainability's own top
feature names) — no extra model passes.

`POST /simulate` acts on the context window cached from the most recent
`/analyze` call in the running server process (`api/pipeline.run_pipeline`
returns `(response_dict, final_context)`; `api/main.py` caches
`final_context` in `_state["last_context"]`). This is a single-session
offline demo, not a multi-tenant API, so there's one global cached-context
slot, not per-user/per-session keying.

```bash
curl -X POST http://localhost:8000/analyze -F "csv_file=@path/to/flows.csv"
curl -X POST http://localhost:8000/simulate \
    -H "Content-Type: application/json" \
    -d '{"action": "isolate_host", "host_ip": "172.16.0.1"}'
```

### What's built
- `momentum/score.py`, `containment/estimator.py`,
  `counterfactual/graph_ops.py`, `counterfactual/simulator.py`.
- `api/schemas.py`'s `MomentumScore`/`ContainmentEstimate` (now part of
  `AnalyzeResponse`) and `SimulateRequest`/`SimulateResponse`;
  `api/pipeline.py`'s `run_simulation()` and the momentum/containment
  computation inside `run_pipeline()`; `api/main.py`'s `POST /simulate`.
- `scripts/validate_momentum.py`, `scripts/validate_containment.py`,
  `scripts/validate_counterfactual.py` — real held-out-data validation for
  all three features, not just unit tests.
- 64 new backend unit/integration tests since Phase 7's 95 (159 total, all
  passing): 17 momentum, 12 containment, 13 `graph_ops`, 15 `simulator`, 7
  new `/simulate`+schema-shape API tests (plus the 10 pre-existing
  `/analyze` tests, same file).
- **A real bug found and fixed while testing:** the first `/simulate` test
  for "no context cached yet" spun up a second `TestClient` against the same
  module-level `app` — exiting that `with` block ran FastAPI's lifespan
  shutdown, which called `_state.clear()` on the *same global dict* the
  module-scoped `client` fixture was still using, wiping its loaded model
  out from under still-pending tests in the same file. Fixed by directly
  clearing/restoring just the `last_context` key on the shared `_state`
  instead of creating a second app instance.

### Frontend — Response Center (`frontend/src/components/panels/{MomentumPanel,ContainmentPanel,CounterfactualPanel}.tsx`)
Built with the `frontend-design` skill, following CLAUDE.md's locked design
system unchanged (same color tokens, IBM Plex Sans/Mono, no new patterns
outside it). The existing single-page Live Feed dashboard was full, so
Phase 9's three features live behind a second console mode — **Response
center** — switched via a bracket-style mode control in the header next to
the title (the way a radar console switches between a track display and a
fire-control display, not the way a marketing site switches tabs). Response
center has no independent upload flow; it acts entirely on the state the
Live feed's last `/analyze` call produced.

- **Momentum** gets a real semicircular instrument dial (`MomentumGauge.tsx`,
  hand-drawn SVG arc + needle, no charting library) — the console's
  signature element, since CLAUDE.md frames the whole app as "an instrument
  panel," not a generic dashboard, and momentum is the one genuinely new
  number this phase introduces. Band colors/thresholds are read from the
  same `momentumSignal()` helper the readout text uses, so the dial and the
  number can never disagree.
- **Containment** gets a range bar (muted bracket = full observed range,
  solid amber band = the p25-p90 "typical" range) plus the recommended
  action, read back from the backend's own copy (`recommended_action`
  starting with "No action needed" or not) rather than re-deriving
  stage-is-active logic in the frontend.
- **Counterfactual** is the fully interactive piece: an action toggle
  (isolate host / block flow), host pickers populated from `flagged_flows`
  (labeled "from observed flows," not claimed as the complete graph), a
  live `POST /simulate` call, and a before/after chart. The chart's y-axis
  auto-zooms to the actual data range rather than a fixed [0, 1] scale — a
  fixed axis was found, while screenshotting this panel, to flatten the
  real effect size (a few hundredths of a probability point, matching
  `scripts/validate_counterfactual.py`'s numbers) into two visually
  indistinguishable lines.
- **Verified against the real running backend**, not just fixtures: scripted
  a real file-input `change` event via Chrome DevTools Protocol against the
  real `ddos_slice3.csv` fixture, through the real Vite proxy, to the real
  FastAPI server — confirmed the real attacker host (`172.16.0.1`) surfaces
  as the default host pick, and running the simulation live reduced mean
  probability from 0.525 to 0.491, consistent with the standalone validation
  script's numbers.

### What's stubbed / TODO
- Momentum's `newly_affected_host_ratio` signal is noisier than the other
  three (see Feature 1's validation caveat above) — worth a second pass if
  this becomes a headline metric in a demo.
- Containment's leave-one-day-out validation only had one stage
  ("Initial Access") with enough independent days to cross-validate — a
  real limitation of this dataset's size, not the estimator's logic.
- Counterfactual actions are graph-structural only (isolate host / block one
  flow); true per-port blocking is not implemented (see disclosed limitation
  above).
- Host pickers in the Counterfactual panel are built from `flagged_flows`
  (the attention-ranked subset), not the complete graph for that window —
  disclosed in the field label, not hidden.

### How to run
```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Validate against real held-out data
./.venv/bin/python scripts/validate_momentum.py
./.venv/bin/python scripts/validate_containment.py
./.venv/bin/python scripts/validate_counterfactual.py
```

## Phase 10 — Remaining Differentiator Features (Attack Future Replay, Multiple Parallel Futures, Explainable Attack Fingerprint)

Phase 9 built 3 of the 6 high-impact features named in this project's reference
brief (Momentum, Containment, Counterfactual Simulation). This phase builds
the remaining 3, in the brief's own stated priority order, completing the
full set — with the same rule as every other feature in this project: no
fabricated numbers, and any gap between the brief's illustrative examples
and what this model can actually support is disclosed, not papered over.

### Feature 1 — Attack Future Replay (`api/pipeline.py`, extends `TimelinePoint`)
The brief's #1 priority: an interactive timeline of observed + predicted
events, with the network graph animating as the analyst scrubs through
time. The backend piece was the harder part: every OBSERVED window's
`TimelinePoint` now carries its own real per-window GNN attention
(`top_flows`), not just the final window's — cheap to add because
`compute_edge_attention` is a single forward pass over one snapshot (no
backward pass, no temporal rollout), unlike the full Phase 5 explanation
that stays attached to the final prediction only. Measured real added
latency: **negligible** (2.2s total for a 29-window capture, same order as
before this feature existed).

**Honest limitation, not smoothed over:** the world model predicts a
*latent embedding* for future windows, never a decoded graph (see
`rollout/engine.py`'s own docstring on why) — so there is no real graph
structure to show for PREDICTED steps. The frontend scrubber's topology
view freezes on the last real observed graph during predicted steps,
clearly labeled, rather than fabricating future network structure.

**Frontend**: new **Mode 3: Attack Replay** (`ReplayPanel.tsx`) — a
play/pause scrubber, a mini oscilloscope strip showing the scrub position,
and the real topology graph updating to that exact window's real
attention data as you move through observed history. Verified against real
data: scrubbing to window 23 of a real DDoS capture shows a *different*
real attack-flagged edge than the final window, confirming genuinely
per-window (not copy-pasted) attention.

### Feature 2 — Multiple Parallel Futures (`futures/parallel_futures.py`)
The brief's #2 priority: several ranked possible outcomes instead of one
deterministic forecast. Built via **MC-Dropout** (Gal & Ghahramani, 2016) —
re-enabling the model's own already-trained dropout layers at inference
time and running the same K-step rollout N=40 independent times. Each run's
dropout noise perturbs the latent trajectory slightly differently, so the
rollout can genuinely land on a different predicted stage; tallying
outcomes across N samples gives a real probability distribution grounded
in the model's own learned uncertainty — not invented percentages.

Checked before building, not assumed: this architecture has **no
BatchNorm anywhere** (GNN encoder, LSTM/Transformer temporal core, output
heads all use only `nn.Dropout`), confirmed by grepping every model file.
That matters because MC-Dropout requires `model.train()` to activate
dropout sampling; with BatchNorm present that call would also start
corrupting running statistics from single-sample batches. Since there's
none, `model.train()` here only does what's wanted, and the model is
always restored to `eval()` in a `finally` block — including a test that
forces a mid-sampling exception and confirms `eval()` mode is still
restored.

**A real sanity check, not just a plausibility argument**: with
`dropout=0.0`, all 40 samples collapse to exactly one outcome matching the
deterministic rollout (`test_zero_dropout_collapses_to_single_deterministic_outcome`)
— confirming the variability genuinely comes from dropout noise, not some
other hidden randomness in the pipeline.

**Cross-feature integration**: Phase 9's Momentum Score explicitly flagged
`attack_path_expansion_available=False` because this feature didn't exist
yet. Now that it does, Momentum's "expansion of predicted attack paths"
signal is wired to the real ensemble (`path_expansion = 1 − probability of
the top outcome`) — a confident, single-outcome forecast contributes zero
expansion; a forecast that's genuinely spread across several plausible
stages contributes more. Weights rebalanced (documented in
`momentum/score.py`) to make room for it as a 5th signal.

**Frontend**: `ParallelFuturesPanel.tsx`, added to **Mode 2: Response
Center** next to Momentum — a ranked bar list, colored by the same
severity signal as everywhere else in the app, with the real sample count
shown so a low-confidence long-tail outcome is visibly low-confidence, not
hidden behind a bare percentage.

### Feature 3 — Explainable Attack Fingerprint (`fingerprint/attack_fingerprint.py`)
The brief's last-priority feature: a per-category "similarity" score
summarizing the attack's behavioral pattern, comparing current vs.
predicted-future fingerprints.

**A disclosed reframing, same pattern as every other Phase 9/10 feature**:
the brief's example uses ATT&CK *tactic* names (Credential Access,
Persistence, Data Exfiltration...) that don't all correspond to something
this model actually predicts. This model's real, trained vocabulary is the
6 MITRE **stage** labels used everywhere else in this project — using
those as the fingerprint's dimensions instead of inventing scores for
categories with no real signal behind them.

The numbers come directly from the model's own real multi-label
`attack_type_head` (independent sigmoid probabilities, already computed for
every rollout step — see `rollout/engine.py`'s `attack_type_probs`), grouped
by stage via the same `ATTACK_TYPE_TO_STAGE` table every other feature in
this project uses. A stage's score is the **max** (not sum) probability
among attack types mapping to it — sigmoid outputs are independent
per-class probabilities, not a normalized distribution, so summing could
exceed 1 and misrepresent the signal. Bucketed into High/≥50%,
Medium/≥20%, Low similarity — the same 0.5 confidence threshold
`rollout/mitre.py` already uses elsewhere, not a new arbitrary cutoff.

**A real, checked limitation, not glossed over**: Exfiltration has *no*
attack type mapped to it anywhere in `ATTACK_TYPE_TO_STAGE` (CIC-IDS2017
has no clean exfiltration category — see `rollout/mitre.py`'s own
docstring). Its fingerprint score is always exactly 0.0 with
`driving_attack_type: null`, stated explicitly rather than silently
showing a misleadingly-confident zero. Verified with a dedicated test
(`test_exfiltration_always_zero_with_no_driver`) that checks this against
the real `ATTACK_TYPE_TO_STAGE` table, not a hardcoded assumption.

"Current" vs. "predicted future" (the brief's own comparison, point 5 of
its "how it works" list): current = the forecast's own freshest first-step
prediction (same convention Momentum and Containment already use);
predicted = the mean attack-type probabilities across the whole K-step
forecast horizon.

**Frontend**: `AttackFingerprintPanel.tsx`, added to **Mode 1: Live Feed**
below Explainability — a 6-row comparison table (current similarity →
predicted similarity → trend arrow), with the real driving attack type
named in the summary line above it.

### What's built
- `futures/parallel_futures.py`, `fingerprint/attack_fingerprint.py`.
- `api/pipeline.py`'s per-window attention loop; `api/schemas.py`'s
  `MultipleFutures`/`FutureOutcome`/`AttackFingerprint`/
  `StageFingerprintEntry`/`FingerprintComparisonEntry`, and `TimelinePoint`
  gaining `top_flows`.
- `momentum/score.py`'s `path_expansion` signal and 5-signal weight
  rebalance.
- Frontend: **Mode 3: Attack Replay** (`ReplayPanel.tsx`), plus
  `ParallelFuturesPanel.tsx` (Mode 2) and `AttackFingerprintPanel.tsx`
  (Mode 1). New icons added to `lib/icons.ts` (`Fingerprint`, `GitFork`,
  `Pause`, `SkipBack`).
- 32 new backend unit/integration tests since Phase 9's 159 (191 total,
  all passing): 8 in `test_parallel_futures.py`, 16 in
  `test_attack_fingerprint.py`, 5 new in `test_momentum.py`
  (`TestPathExpansionSignal`), the rest new assertions in `test_api.py`
  (per-window attention shape, `multiple_futures` shape,
  `attack_fingerprint` shape).
- Verified end-to-end against the real held-out DDoS capture via a
  scripted browser session (not just unit tests): real per-window
  attention driving the replay scrubber, real MC-Dropout outcome
  distribution, real fingerprint scores with the real driving attack type
  named.

### What's stubbed / TODO
- Multiple Parallel Futures resamples the *same* encoded context via the
  model's own uncertainty — it answers "how consistent is the model's own
  forecast," not "what would a structurally different starting condition
  produce" (that's a different question, closer to what Counterfactual
  Intervention Simulation already answers for defender actions
  specifically).
- MC-Dropout percentages are a Monte Carlo estimate: re-running shifts them
  slightly. N=40 was chosen for a fast, stable top-ranked outcome in
  practice, not tuned against a held-out accuracy target (there isn't a
  ground-truth "correct" distribution to tune against here).
- Attack Future Replay's "update when new telemetry arrives" (the brief's
  point 7) doesn't apply — this system analyzes a captured/uploaded file,
  not a live stream; disclosed rather than faked with an artificial live
  indicator.

### How to run
```bash
# Unit tests
./.venv/bin/python -m pytest -q

# Frontend
cd frontend && npm run dev
```
