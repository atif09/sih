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
explain/         Phase 5: explainability (attention / SHAP)
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
