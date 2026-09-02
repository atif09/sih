"""
Packet-level feature extraction from raw PCAP files, via Scapy.

STATUS: built and unit-tested against synthetically crafted packets (see
tests/test_packet_features.py). NOT yet run against real captured traffic --
the user has not provided PCAP files as of Phase 1 (data/cicids2018-pcap/ and
data/ctu13/<scenario>/ are empty placeholders). Point this module at real PCAPs
via `extract_packet_features()` once they're available; flag any schema
surprises (e.g. VLAN tags, IPv6, tunneling) back for a schema update.

Why packet-level, in addition to flow-level (flow_features.py)
----------------------------------------------------------------
CICFlowMeter's flow-level output is an aggregate: it tells you a flow had say
40 packets and 1200 bytes, but not *how* those packets were shaped or timed
within the flow. That aggregation is exactly what hides a slow, low-and-slow
reconnaissance scan (a handful of packets per target, spread thin) from
flow-level thresholds. This module recovers that sub-flow signal:
    - TTL values + variance (spoofed/relayed traffic often has anomalous or
      inconsistent TTL within what should be one path)
    - TCP window size (announces receive-buffer behavior; scanning tools and
      some OS TCP stacks use characteristic/static window sizes)
    - IP fragmentation flags (fragmentation is unusual for ordinary flows and
      is a known evasion technique)
    - Payload size distribution (per-packet, not just flow total)
    - Port scan signatures (sequential vs. randomized destination-port access
      by a single source, within a time window)
    - Retransmission counts (repeated TCP sequence numbers in one direction)

Flow reconstruction
--------------------
Raw packets are not pre-grouped into flows the way CICFlowMeter's output is, so
this module groups packets into flows itself: a canonical 5-tuple key (with
endpoints ordered so both directions of a conversation map to the same key),
split into a new flow instance whenever the gap between consecutive packets on
that key exceeds `flow_timeout_seconds` (default 120s, matching CICFlowMeter's
own default flow timeout, so flow boundaries are comparable across the two
extractors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IP "More Fragments" flag bit and a nonzero fragment offset both indicate this
# packet is part of a fragmented datagram.
IP_FLAG_MF = 0x1


def _require_scapy():
    try:
        import scapy.all as scapy  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scapy is required for packet-level feature extraction. "
            "Install project requirements: pip install -r requirements.txt"
        ) from exc
    return scapy


@dataclass
class _FlowAccumulator:
    """Mutable state accumulated while scanning packets for a single flow
    instance (one 5-tuple, one contiguous burst of activity)."""

    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: int
    start_ts: float
    last_ts: float = 0.0
    packet_count: int = 0
    ttls: list[int] = field(default_factory=list)
    tcp_windows: list[int] = field(default_factory=list)
    frag_count: int = 0
    payload_sizes: list[int] = field(default_factory=list)
    # Sequence numbers seen per direction ("fwd" = matches src_ip/src_port of
    # the packet that opened this flow). A seq number repeated within a
    # direction (and not a keepalive / zero-length ack) is a retransmission.
    seen_seqs: dict[str, set[int]] = field(default_factory=lambda: {"fwd": set(), "bwd": set()})
    retransmissions: int = 0

    def direction(self, pkt_src_ip: str, pkt_src_port: int) -> str:
        return "fwd" if (pkt_src_ip, pkt_src_port) == (self.src_ip, self.src_port) else "bwd"

    def to_record(self) -> dict:
        ttl_var = float(np.var(self.ttls)) if len(self.ttls) > 1 else 0.0
        win_var = float(np.var(self.tcp_windows)) if len(self.tcp_windows) > 1 else 0.0
        payload_arr = np.asarray(self.payload_sizes, dtype=np.float64)
        return {
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "start_ts": self.start_ts,
            "end_ts": self.last_ts,
            "packet_count": self.packet_count,
            "ttl_mean": float(np.mean(self.ttls)) if self.ttls else 0.0,
            "ttl_var": ttl_var,
            "ttl_min": float(np.min(self.ttls)) if self.ttls else 0.0,
            "ttl_max": float(np.max(self.ttls)) if self.ttls else 0.0,
            "tcp_window_mean": float(np.mean(self.tcp_windows)) if self.tcp_windows else 0.0,
            "tcp_window_var": win_var,
            "ip_frag_count": self.frag_count,
            "ip_frag_ratio": self.frag_count / self.packet_count if self.packet_count else 0.0,
            "payload_size_mean": float(payload_arr.mean()) if payload_arr.size else 0.0,
            "payload_size_std": float(payload_arr.std()) if payload_arr.size else 0.0,
            "payload_size_min": float(payload_arr.min()) if payload_arr.size else 0.0,
            "payload_size_max": float(payload_arr.max()) if payload_arr.size else 0.0,
            "retransmission_count": self.retransmissions,
        }


def _canonical_key(ip_a: str, port_a: int, ip_b: str, port_b: int, proto: int):
    """Order the two endpoints deterministically so both directions of one
    conversation hash to the same flow key."""
    ep_a, ep_b = (ip_a, port_a), (ip_b, port_b)
    return (ep_a, ep_b, proto) if ep_a <= ep_b else (ep_b, ep_a, proto)


def iter_ip_packets(pcap_path: str | Path) -> Iterator[tuple]:
    """Stream (timestamp, scapy_packet) pairs for every IP packet in a pcap,
    without loading the whole capture into memory. Non-IP packets (ARP, etc.)
    are skipped."""
    scapy = _require_scapy()
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(
            f"{pcap_path} does not exist. Place raw PCAP day-folders in "
            f"data/cicids2018-pcap/<day-name>/ or data/ctu13/<scenario>/."
        )
    with scapy.PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if scapy.IP in pkt:
                ts = float(pkt.time)
                yield ts, pkt


def extract_packet_features(
    pcap_path: str | Path,
    flow_timeout_seconds: float = 120.0,
) -> pd.DataFrame:
    """Parse a PCAP and return one row per reconstructed flow instance with the
    packet-level features described in the module docstring. Flow instances are
    split on `flow_timeout_seconds` of inactivity, matching CICFlowMeter's
    default so flow boundaries roughly line up with flow_features.py output.
    """
    scapy = _require_scapy()

    # key -> currently-open _FlowAccumulator for that 5-tuple.
    open_flows: dict[tuple, _FlowAccumulator] = {}
    finished: list[dict] = []

    # IP fragmentation reassembly note: only the FIRST fragment of a
    # fragmented datagram carries the L4 (TCP/UDP) header -- Scapy correctly
    # declines to dissect one on later fragments, matching real fragmented
    # traffic, so those packets have no port info to key a flow on. We
    # recover the correlation via (src_ip, dst_ip, protocol, IP identification
    # field), which uniquely identifies "later fragments belonging to this
    # datagram" and lets us attribute them to the flow the first fragment
    # opened, instead of bucketing them into a bogus port-0 pseudo-flow.
    frag_id_to_flow_key: dict[tuple, tuple] = {}

    n_packets = 0
    for ts, pkt in iter_ip_packets(pcap_path):
        n_packets += 1
        ip_layer = pkt[scapy.IP]
        proto = int(ip_layer.proto)
        ttl = int(ip_layer.ttl)
        is_fragmented = bool(int(ip_layer.flags) & IP_FLAG_MF) or int(ip_layer.frag) > 0
        frag_datagram_id = (ip_layer.src, ip_layer.dst, proto, int(ip_layer.id))

        has_tcp = scapy.TCP in pkt
        has_udp = scapy.UDP in pkt
        if has_tcp:
            sport, dport = int(pkt[scapy.TCP].sport), int(pkt[scapy.TCP].dport)
        elif has_udp:
            sport, dport = int(pkt[scapy.UDP].sport), int(pkt[scapy.UDP].dport)
        elif is_fragmented and frag_datagram_id in frag_id_to_flow_key:
            # A later fragment of a datagram we've already seen the first
            # fragment of -- reuse that flow's key rather than falling back
            # to an uncorrelated port-0 key.
            key = frag_id_to_flow_key[frag_datagram_id]
            sport, dport = key[0][1], key[1][1]
        else:
            sport, dport = 0, 0

        key = _canonical_key(ip_layer.src, sport, ip_layer.dst, dport, proto)
        if is_fragmented and (has_tcp or has_udp):
            # This is the first fragment (it still carries an L4 header) --
            # remember its flow key so later fragments of the same datagram
            # can be attributed correctly above.
            frag_id_to_flow_key[frag_datagram_id] = key

        flow = open_flows.get(key)

        if flow is not None and (ts - flow.last_ts) > flow_timeout_seconds:
            finished.append(flow.to_record())
            flow = None

        if flow is None:
            flow = _FlowAccumulator(
                src_ip=ip_layer.src, src_port=sport,
                dst_ip=ip_layer.dst, dst_port=dport,
                protocol=proto, start_ts=ts, last_ts=ts,
            )
            open_flows[key] = flow

        flow.last_ts = ts
        flow.packet_count += 1
        flow.ttls.append(ttl)

        if is_fragmented:
            flow.frag_count += 1

        payload_len = len(bytes(ip_layer.payload))
        if has_tcp:
            tcp = pkt[scapy.TCP]
            flow.tcp_windows.append(int(tcp.window))
            payload_len = len(bytes(tcp.payload))
            direction = flow.direction(ip_layer.src, sport)
            seq = int(tcp.seq)
            # A zero-payload ACK/keepalive can legitimately repeat a seq
            # number; only count it as a retransmission if it actually
            # carried data (a real resend), avoiding false positives on
            # ordinary ACK traffic.
            if payload_len > 0:
                if seq in flow.seen_seqs[direction]:
                    flow.retransmissions += 1
                else:
                    flow.seen_seqs[direction].add(seq)
        flow.payload_sizes.append(payload_len)

    for flow in open_flows.values():
        finished.append(flow.to_record())

    if n_packets == 0:
        logger.warning("%s: no IP packets found", pcap_path)

    if not finished:
        return pd.DataFrame(columns=[
            "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
            "start_ts", "end_ts", "packet_count", "ttl_mean", "ttl_var",
            "ttl_min", "ttl_max", "tcp_window_mean", "tcp_window_var",
            "ip_frag_count", "ip_frag_ratio", "payload_size_mean",
            "payload_size_std", "payload_size_min", "payload_size_max",
            "retransmission_count",
        ])
    return pd.DataFrame(finished)


def detect_port_scan_signatures(
    flow_df: pd.DataFrame,
    window_size_seconds: float = 60.0,
) -> pd.DataFrame:
    """Given the per-flow output of extract_packet_features, detect port-scan
    signatures per (src_ip, time window): a single source contacting many
    distinct destination ports (optionally across many destination IPs) within
    a short window is scan-shaped traffic, regardless of whether any single
    flow crossed a volumetric threshold.

    Two scan styles are distinguished, per the PS's wording ("sequential or
    randomised port access patterns"):
      - sequential: distinct ports, sorted, mostly consecutive (e.g. 1,2,3,4..)
      - randomized: many distinct ports with no consecutive structure

    Returns one row per (src_ip, window_start) with:
        unique_dst_ports, unique_dst_ips, port_sequential_ratio,
        is_sequential_scan, is_randomized_scan, port_scan_score
    """
    if flow_df.empty:
        return pd.DataFrame(columns=[
            "src_ip", "window_start", "unique_dst_ports", "unique_dst_ips",
            "port_sequential_ratio", "is_sequential_scan",
            "is_randomized_scan", "port_scan_score",
        ])

    df = flow_df.copy()
    df["window_start"] = (df["start_ts"] // window_size_seconds) * window_size_seconds

    records = []
    for (src_ip, window_start), group in df.groupby(["src_ip", "window_start"]):
        ports = sorted(set(int(p) for p in group["dst_port"]))
        n_ports = len(ports)
        n_ips = group["dst_ip"].nunique()

        if n_ports > 1:
            diffs = np.diff(ports)
            sequential_ratio = float(np.mean(diffs == 1))
        else:
            sequential_ratio = 0.0

        # Heuristic thresholds (documented, not learned): >=4 distinct ports
        # touched by one source in one window is the minimum signal we treat
        # as "scan-shaped" at all; above that, we classify style by how
        # consecutive the port sequence is.
        is_scan_shaped = n_ports >= 4
        is_sequential = is_scan_shaped and sequential_ratio >= 0.5
        is_randomized = is_scan_shaped and sequential_ratio < 0.5

        # Score in [0, 1]: driven by breadth of ports touched relative to how
        # many packets it took to touch them (a scan is "wide and shallow").
        avg_packets_per_port = group["packet_count"].sum() / n_ports if n_ports else 0.0
        breadth = min(n_ports / 20.0, 1.0)  # saturate at 20 distinct ports
        shallowness = 1.0 / (1.0 + avg_packets_per_port)
        score = breadth * shallowness if is_scan_shaped else 0.0

        records.append({
            "src_ip": src_ip,
            "window_start": window_start,
            "unique_dst_ports": n_ports,
            "unique_dst_ips": int(n_ips),
            "port_sequential_ratio": sequential_ratio,
            "is_sequential_scan": is_sequential,
            "is_randomized_scan": is_randomized,
            "port_scan_score": float(score),
        })

    return pd.DataFrame(records)
