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
