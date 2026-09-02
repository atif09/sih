# SIH26153 — AI-Based Network Attack Forecasting (World Models)

World-model-based system that learns network state transition dynamics to
forecast attacker progression before compromise completes. Built for SIH 2026,
PS 26153.

## Project layout

```
data/            Dataset storage (cicids2018/, cicids2018-pcap/, ctu13/) + data-loading code
features/        Phase 1: flow-level + packet-level feature extraction, windowing/merge
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
