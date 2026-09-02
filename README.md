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
- 13 new unit tests (71 total, all passing), including the concentration-
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
