"""
Graph state representation: one GraphSnapshot per time window, built from
Phase 1's merged/windowed flow-level feature table.

Design decisions (documented here since the PS leaves the exact
representation open)
---------------------------------------------------------------------------
- **Nodes = hosts (IP addresses)** observed as either the source or
  destination of at least one flow within the window. A host that sent or
  received nothing in a window simply doesn't appear in that window's graph
  (variable-size graphs across time are expected and are what the GNN
  encoder in Phase 3 is for).

- **Edges = one aggregated directed edge per (src_ip, dst_ip) ordered pair
  per window**, not one edge per raw flow. A host pair can have many
  individual flows in one window (e.g. a port scan touches the same target
  on 30 different ports); collapsing them into one edge with aggregated
  statistics (flow_count, byte/packet sums, flag counts, distinct-port
  counts, etc.) keeps the graph size proportional to the number of *host
  relationships* active in a window rather than the number of packets, which
  is both more tractable for a GNN and a more natural definition of "network
  state" (the PS's own framing: "active flows" between hosts, not individual
  packets). Directionality (src -> dst) is preserved since attacker ->
  target is directionally meaningful for stage mapping.

- **Node/edge features are aggregated from Phase 1's per-flow columns**,
  which already include packet-level + port-scan signal when a PCAP was
  available (zero-filled otherwise, per Phase 1's `packet_features_available`
  flag). No feature engineering is redone here -- this module is purely an
  aggregation from flow-grain to host-grain and host-pair-grain.

- **Labels are kept separate from features.** `node_labels` / `edge_labels`
  / `graph_attack_present` are derived from the ground-truth Label column and
  must never be used as model *inputs* (that would leak the answer) --
  they're the training targets, consumed by graph/transitions.py.

- **Features are NOT normalized here.** Phase 3 will hold out a full
  day/attack-type for generalization testing; fitting a scaler before the
  train/val/test split would leak information from the held-out day into the
  training distribution. This module outputs raw aggregated values;
  normalization happens once, after the split, in Phase 3.

- Node ids are kept as the actual IP address strings (not just a dense local
  index) throughout, so any downstream use that wants to track a specific
  host's behavior over time (e.g. Phase 5 explainability, or a future
  per-node-recurrent architecture) can do so without re-deriving identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Node label encoding. A host can simultaneously be the source of one
# malicious edge and the target of another within the same window (e.g. a
# compromised pivot host during lateral movement), hence the BOTH state.
NODE_LABEL_BENIGN = 0
NODE_LABEL_ATTACK_SOURCE = 1
NODE_LABEL_ATTACK_TARGET = 2
NODE_LABEL_ATTACK_BOTH = 3

# Fixed, positionally-meaningful feature name lists. Every GraphSnapshot
# (across every window, every day/file) uses exactly these columns in this
# order, so snapshots can be concatenated/batched downstream without
# per-snapshot schema bookkeeping.
NODE_FEATURE_NAMES: tuple[str, ...] = (
    "out_flow_count", "out_unique_peers", "out_unique_ports",
    "out_bytes_sum", "out_bytes_mean", "out_pkts_sum", "out_pkts_mean",
    "out_duration_mean", "out_iat_mean_mean",
    "out_syn_cnt_sum", "out_ack_cnt_sum", "out_fin_cnt_sum",
    "out_rst_cnt_sum", "out_psh_cnt_sum", "out_urg_cnt_sum",
    "out_port_scan_score_max", "out_ttl_mean_mean", "out_ttl_var_mean",
    "out_retransmission_sum",
    "in_flow_count", "in_unique_peers",
    "in_bytes_sum", "in_bytes_mean", "in_pkts_sum", "in_pkts_mean",
    "in_duration_mean", "in_iat_mean_mean",
    "in_syn_cnt_sum", "in_ack_cnt_sum", "in_fin_cnt_sum",
    "in_rst_cnt_sum", "in_psh_cnt_sum", "in_urg_cnt_sum",
    "in_ttl_mean_mean", "in_ttl_var_mean", "in_retransmission_sum",
    "total_degree", "packet_features_available_frac",
)

EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "flow_count", "bytes_sum", "bytes_mean", "pkts_sum", "pkts_mean",
    "duration_mean", "iat_mean_mean",
    "syn_cnt_sum", "ack_cnt_sum", "fin_cnt_sum", "rst_cnt_sum",
    "psh_cnt_sum", "urg_cnt_sum", "flag_bitmask_or",
    "down_up_ratio_mean", "fwd_bwd_byte_ratio_mean", "dst_port_nunique",
    "ttl_mean_mean", "ttl_var_mean", "tcp_window_mean_mean",
    "ip_frag_ratio_mean", "payload_size_mean_mean", "retransmission_sum",
    "packet_features_available_frac",
)


@dataclass
class GraphSnapshot:
    day: str
    window_id: int
    window_start: float

    node_ids: list[str]
    node_features: np.ndarray  # (N, len(NODE_FEATURE_NAMES)), float64
    node_labels: np.ndarray  # (N,), int64, one of NODE_LABEL_*

    edge_index: np.ndarray  # (2, E), int64 -- row0=src idx, row1=dst idx into node_ids
    edge_features: np.ndarray  # (E, len(EDGE_FEATURE_NAMES)), float64
    edge_labels: np.ndarray  # (E,), int64, 0/1
    edge_attack_types: list[frozenset]  # length E, set of non-benign label strings per edge

    graph_attack_present: bool
    graph_attack_types: frozenset = field(default_factory=frozenset)
    num_flows: int = 0  # raw flow rows aggregated into this snapshot (0 for an empty/quiet window)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def empty_like(self) -> bool:
        return self.num_nodes == 0


def _empty_snapshot(day: str, window_id: int, window_start: float) -> GraphSnapshot:
    """A quiet window: no flows observed at all. Represented as a valid
    zero-node graph rather than skipped, so fixed-Δt sequence models see a
    real (if empty) state instead of a silent gap in time."""
    return GraphSnapshot(
        day=day, window_id=window_id, window_start=window_start,
        node_ids=[], node_features=np.zeros((0, len(NODE_FEATURE_NAMES))),
        node_labels=np.zeros((0,), dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_features=np.zeros((0, len(EDGE_FEATURE_NAMES))),
        edge_labels=np.zeros((0,), dtype=np.int64),
        edge_attack_types=[],
        graph_attack_present=False, graph_attack_types=frozenset(), num_flows=0,
    )


def _aggregate_edges(window_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (window_id, src_ip, dst_ip), aggregated from all flows
    sharing that ordered host pair within the window."""

    def _flag_or(s: pd.Series) -> int:
        result = 0
        for v in s:
            result |= int(v)
        return result

    def _attack_types(s: pd.Series) -> frozenset:
        # Real (unlabeled) inference-time uploads have no Label column --
        # flow_features.py sets label to NaN in that case, not the string
        # "Benign". Filtering only "benign" (case-insensitively) would let
        # NaN through and produce a spurious attack-type literally named
        # "nan" (str(float('nan')) == "nan"). pd.isna() catches both.
        return frozenset(
            v for v in s.unique()
            if not pd.isna(v) and str(v).strip().lower() != "benign"
        )

    agg = window_df.groupby(["window_id", "src_ip", "dst_ip"]).agg(
        flow_count=("label", "size"),
        bytes_sum=("tot_bytes", "sum"), bytes_mean=("tot_bytes", "mean"),
        pkts_sum=("tot_pkts", "sum"), pkts_mean=("tot_pkts", "mean"),
        duration_mean=("duration_us", "mean"),
        iat_mean_mean=("flow_iat_mean", "mean"),
        syn_cnt_sum=("syn_flag_cnt", "sum"), ack_cnt_sum=("ack_flag_cnt", "sum"),
        fin_cnt_sum=("fin_flag_cnt", "sum"), rst_cnt_sum=("rst_flag_cnt", "sum"),
        psh_cnt_sum=("psh_flag_cnt", "sum"), urg_cnt_sum=("urg_flag_cnt", "sum"),
        flag_bitmask_or=("flag_bitmask", _flag_or),
        down_up_ratio_mean=("down_up_ratio", "mean"),
        fwd_bwd_byte_ratio_mean=("fwd_bwd_byte_ratio", "mean"),
        dst_port_nunique=("dst_port", "nunique"),
        ttl_mean_mean=("ttl_mean", "mean"), ttl_var_mean=("ttl_var", "mean"),
        tcp_window_mean_mean=("tcp_window_mean", "mean"),
        ip_frag_ratio_mean=("ip_frag_ratio", "mean"),
        payload_size_mean_mean=("payload_size_mean", "mean"),
        retransmission_sum=("retransmission_count", "sum"),
        packet_features_available_frac=("packet_features_available", "mean"),
        edge_label=("label_binary", "max"),
        attack_types=("label", _attack_types),
    ).reset_index()
    return agg


def _aggregate_nodes(window_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (window_id, host_ip), combining that host's behavior as a
    flow source ("out") and as a flow destination ("in") within the window."""
    out_agg = window_df.groupby(["window_id", "src_ip"]).agg(
        out_flow_count=("label", "size"),
        out_unique_peers=("dst_ip", "nunique"),
        out_unique_ports=("dst_port", "nunique"),
        out_bytes_sum=("tot_bytes", "sum"), out_bytes_mean=("tot_bytes", "mean"),
        out_pkts_sum=("tot_pkts", "sum"), out_pkts_mean=("tot_pkts", "mean"),
        out_duration_mean=("duration_us", "mean"),
        out_iat_mean_mean=("flow_iat_mean", "mean"),
        out_syn_cnt_sum=("syn_flag_cnt", "sum"), out_ack_cnt_sum=("ack_flag_cnt", "sum"),
        out_fin_cnt_sum=("fin_flag_cnt", "sum"), out_rst_cnt_sum=("rst_flag_cnt", "sum"),
        out_psh_cnt_sum=("psh_flag_cnt", "sum"), out_urg_cnt_sum=("urg_flag_cnt", "sum"),
        out_port_scan_score_max=("port_scan_score", "max"),
        out_ttl_mean_mean=("ttl_mean", "mean"), out_ttl_var_mean=("ttl_var", "mean"),
        out_retransmission_sum=("retransmission_count", "sum"),
        is_attack_source=("label_binary", "max"),
        pkt_avail_out=("packet_features_available", "mean"),
    ).reset_index().rename(columns={"src_ip": "host_ip"})

    in_agg = window_df.groupby(["window_id", "dst_ip"]).agg(
        in_flow_count=("label", "size"),
        in_unique_peers=("src_ip", "nunique"),
        in_bytes_sum=("tot_bytes", "sum"), in_bytes_mean=("tot_bytes", "mean"),
        in_pkts_sum=("tot_pkts", "sum"), in_pkts_mean=("tot_pkts", "mean"),
        in_duration_mean=("duration_us", "mean"),
        in_iat_mean_mean=("flow_iat_mean", "mean"),
        in_syn_cnt_sum=("syn_flag_cnt", "sum"), in_ack_cnt_sum=("ack_flag_cnt", "sum"),
        in_fin_cnt_sum=("fin_flag_cnt", "sum"), in_rst_cnt_sum=("rst_flag_cnt", "sum"),
        in_psh_cnt_sum=("psh_flag_cnt", "sum"), in_urg_cnt_sum=("urg_flag_cnt", "sum"),
        in_ttl_mean_mean=("ttl_mean", "mean"), in_ttl_var_mean=("ttl_var", "mean"),
        in_retransmission_sum=("retransmission_count", "sum"),
        is_attack_target=("label_binary", "max"),
        pkt_avail_in=("packet_features_available", "mean"),
    ).reset_index().rename(columns={"dst_ip": "host_ip"})

    merged = out_agg.merge(in_agg, on=["window_id", "host_ip"], how="outer")

    count_cols = [c for c in merged.columns if c not in ("window_id", "host_ip")]
    merged[count_cols] = merged[count_cols].fillna(0.0)

    merged["total_degree"] = merged["out_unique_peers"] + merged["in_unique_peers"]
    merged["packet_features_available_frac"] = (
        (merged["pkt_avail_out"] * merged["out_flow_count"]
         + merged["pkt_avail_in"] * merged["in_flow_count"])
        / (merged["out_flow_count"] + merged["in_flow_count"]).replace(0, np.nan)
    ).fillna(0.0)
    merged["is_attack_source"] = merged["is_attack_source"].astype(np.int64)
    merged["is_attack_target"] = merged["is_attack_target"].astype(np.int64)

    return merged


def build_snapshots_for_day(
    windowed_df: pd.DataFrame, day: str, window_size_seconds: float = 60.0,
) -> list[GraphSnapshot]:
    """Build one GraphSnapshot per time window for a single day/file's merged,
    windowed flow-level DataFrame (the output of
    features.windowing.merge_flow_and_packet_features).

    Windows are gap-filled: every integer window_id from the observed min to
    max is materialized, including windows with zero flows (as an explicit
    empty graph), so a fixed-Δt sequence model sees uniform time steps rather
    than silently skipping quiet periods.

    `window_size_seconds` must match the value used to produce `windowed_df`
    (configs/pipeline.yaml -> feature_extraction.window_size_seconds) -- it's
    needed to compute `window_start` for the empty windows this function
    fabricates. It is NOT inferred from `windowed_df` itself: with sparse
    gaps (exactly the case gap-filling handles), the observed window_start
    values can be many steps apart, making the true step ambiguous from data
    alone.
    """
    if windowed_df.empty:
        return []

    windowed_df = windowed_df.copy()
    if "label_binary" in windowed_df.columns:
        # Real (unlabeled) inference-time uploads have no ground truth --
        # flow_features.py leaves label_binary as NaN in that case (see its
        # docstring). Casting NaN straight to int64 later (edge_label,
        # is_attack_source/target) is undefined behavior in numpy -- it does
        # NOT reliably raise; it has been observed to silently produce a
        # platform/version-dependent garbage integer, which could make
        # graph_attack_present spuriously True from a cast artifact rather
        # than real signal. Filled explicitly to 0 here: for unlabeled data
        # this makes graph_attack_present/edge_labels/node_labels a
        # meaningless placeholder (always "no attack"), not real ground
        # truth -- callers doing pure inference (api/, rollout/) never rely
        # on these fields anyway; only training (Phase 3) needs them to be
        # real, and training data always has genuine labels.
        windowed_df["label_binary"] = windowed_df["label_binary"].fillna(0)
    windowed_df["dst_ip"] = windowed_df["dst_ip"].astype(str)

    edge_agg = _aggregate_edges(windowed_df)
    node_agg = _aggregate_nodes(windowed_df)

    window_start_by_id = windowed_df.groupby("window_id")["window_start"].first().to_dict()
    min_wid, max_wid = int(windowed_df["window_id"].min()), int(windowed_df["window_id"].max())
    step = float(window_size_seconds)

    edge_groups = dict(tuple(edge_agg.groupby("window_id")))
    node_groups = dict(tuple(node_agg.groupby("window_id")))

    snapshots: list[GraphSnapshot] = []
    for wid in range(min_wid, max_wid + 1):
        w_start = window_start_by_id.get(wid, window_start_by_id[min_wid] + (wid - min_wid) * step)

        if wid not in node_groups:
            snapshots.append(_empty_snapshot(day, wid, w_start))
            continue

        nodes = node_groups[wid].reset_index(drop=True)
        edges = edge_groups.get(wid, edge_agg.iloc[0:0])

        node_ids = nodes["host_ip"].tolist()
        node_index = {ip: i for i, ip in enumerate(node_ids)}
        node_features = nodes[list(NODE_FEATURE_NAMES)].to_numpy(dtype=np.float64)

        node_labels = np.zeros(len(node_ids), dtype=np.int64)
        is_src = nodes["is_attack_source"].to_numpy()
        is_dst = nodes["is_attack_target"].to_numpy()
        node_labels[(is_src == 1) & (is_dst == 0)] = NODE_LABEL_ATTACK_SOURCE
        node_labels[(is_src == 0) & (is_dst == 1)] = NODE_LABEL_ATTACK_TARGET
        node_labels[(is_src == 1) & (is_dst == 1)] = NODE_LABEL_ATTACK_BOTH

        if len(edges):
            src_idx = edges["src_ip"].map(node_index).to_numpy(dtype=np.int64)
            dst_idx = edges["dst_ip"].map(node_index).to_numpy(dtype=np.int64)
            edge_index = np.vstack([src_idx, dst_idx])
            edge_features = edges[list(EDGE_FEATURE_NAMES)].to_numpy(dtype=np.float64)
            edge_labels = edges["edge_label"].to_numpy(dtype=np.int64)
            edge_attack_types = list(edges["attack_types"])
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, len(EDGE_FEATURE_NAMES)))
            edge_labels = np.zeros((0,), dtype=np.int64)
            edge_attack_types = []

        graph_attack_present = bool(edge_labels.sum() > 0)
        graph_attack_types: frozenset = frozenset().union(*edge_attack_types) if edge_attack_types else frozenset()

        snapshots.append(GraphSnapshot(
            day=day, window_id=wid, window_start=float(w_start),
            node_ids=node_ids, node_features=node_features, node_labels=node_labels,
            edge_index=edge_index, edge_features=edge_features, edge_labels=edge_labels,
            edge_attack_types=edge_attack_types,
            graph_attack_present=graph_attack_present, graph_attack_types=graph_attack_types,
            num_flows=int(nodes["out_flow_count"].sum()),
        ))

    return snapshots
