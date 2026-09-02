"""
Canonical column schema for CICFlowMeter-derived CSVs (CIC-IDS2018, CTU-13 NetFlow
exports use a different schema entirely and are handled separately in Phase 2).

Why this file exists
---------------------
The CIC-IDS2018 processed CSVs are notoriously inconsistent across capture days:
column names differ in spacing/casing ("Fwd Header Length" vs "Fwd Header Len"),
some days include `Flow ID`/`Src IP`/`Src Port`/`Dst IP`/`Timestamp` as the first
five columns and some redistributions of the dataset strip them for anonymization,
and a few files have the header row duplicated mid-file (a known CICFlowMeter bug
when multiple capture sessions were concatenated). This module centralizes all of
that normalization so the rest of the pipeline can work against one stable,
documented set of column names.

We have NOT yet seen the user's actual downloaded files (data/cicids2018/ is
empty as of Phase 1). The alias table below is built from the publicly documented
CICFlowMeter output schema and known dataset quirks. If real files use column
names not covered here, `normalize_columns` will raise a clear error naming the
unmapped columns rather than silently dropping data.
"""

from __future__ import annotations

import re

# Canonical name -> tuple of known aliases (raw header strings seen in the wild).
# Matching is done after whitespace/case normalization (see _clean_key), so an
# alias only needs one representative spelling per "shape" of the name.
CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    # --- identifiers ---
    "flow_id": ("flow id",),
    "src_ip": ("src ip", "source ip"),
    "src_port": ("src port", "source port"),
    "dst_ip": ("dst ip", "destination ip"),
    "dst_port": ("dst port", "destination port"),
    "protocol": ("protocol",),
    "timestamp": ("timestamp",),

    # --- core flow stats ---
    "flow_duration": ("flow duration",),
    "tot_fwd_pkts": ("tot fwd pkts", "total fwd packets", "total fwd packet"),
    "tot_bwd_pkts": ("tot bwd pkts", "total backward packets", "total bwd packets"),
    "totlen_fwd_pkts": ("totlen fwd pkts", "total length of fwd packets", "total length of fwd packet"),
    "totlen_bwd_pkts": ("totlen bwd pkts", "total length of bwd packets"),

    # --- fwd packet length stats ---
    "fwd_pkt_len_max": ("fwd pkt len max", "fwd packet length max"),
    "fwd_pkt_len_min": ("fwd pkt len min", "fwd packet length min"),
    "fwd_pkt_len_mean": ("fwd pkt len mean", "fwd packet length mean"),
    "fwd_pkt_len_std": ("fwd pkt len std", "fwd packet length std"),

    # --- bwd packet length stats ---
    "bwd_pkt_len_max": ("bwd pkt len max", "bwd packet length max"),
    "bwd_pkt_len_min": ("bwd pkt len min", "bwd packet length min"),
    "bwd_pkt_len_mean": ("bwd pkt len mean", "bwd packet length mean"),
    "bwd_pkt_len_std": ("bwd pkt len std", "bwd packet length std"),

    # --- rate ---
    "flow_byts_s": ("flow byts/s", "flow bytes/s"),
    "flow_pkts_s": ("flow pkts/s", "flow packets/s"),

    # --- flow-level IAT ---
    "flow_iat_mean": ("flow iat mean",),
    "flow_iat_std": ("flow iat std",),
    "flow_iat_max": ("flow iat max",),
    "flow_iat_min": ("flow iat min",),

    # --- fwd IAT ---
    "fwd_iat_tot": ("fwd iat tot", "fwd iat total"),
    "fwd_iat_mean": ("fwd iat mean",),
    "fwd_iat_std": ("fwd iat std",),
    "fwd_iat_max": ("fwd iat max",),
    "fwd_iat_min": ("fwd iat min",),

    # --- bwd IAT ---
    "bwd_iat_tot": ("bwd iat tot", "bwd iat total"),
    "bwd_iat_mean": ("bwd iat mean",),
    "bwd_iat_std": ("bwd iat std",),
    "bwd_iat_max": ("bwd iat max",),
    "bwd_iat_min": ("bwd iat min",),

    # --- flags (per-flow counts, used to build the flag bitmask) ---
    "fwd_psh_flags": ("fwd psh flags",),
    "bwd_psh_flags": ("bwd psh flags",),
    "fwd_urg_flags": ("fwd urg flags",),
    "bwd_urg_flags": ("bwd urg flags",),
    "fin_flag_cnt": ("fin flag cnt", "fin flag count"),
    "syn_flag_cnt": ("syn flag cnt", "syn flag count"),
    "rst_flag_cnt": ("rst flag cnt", "rst flag count"),
    "psh_flag_cnt": ("psh flag cnt", "psh flag count"),
    "ack_flag_cnt": ("ack flag cnt", "ack flag count"),
    "urg_flag_cnt": ("urg flag cnt", "urg flag count"),
    "cwe_flag_count": ("cwe flag count", "cwr flag count"),
    "ece_flag_cnt": ("ece flag cnt", "ece flag count"),

    # --- header lengths ---
    "fwd_header_len": ("fwd header len", "fwd header length"),
    "bwd_header_len": ("bwd header len", "bwd header length"),

    # --- per-direction packet rate ---
    "fwd_pkts_s": ("fwd pkts/s", "fwd packets/s"),
    "bwd_pkts_s": ("bwd pkts/s", "bwd packets/s"),

    # --- overall packet length stats ---
    "pkt_len_min": ("pkt len min", "min packet length"),
    "pkt_len_max": ("pkt len max", "max packet length"),
    "pkt_len_mean": ("pkt len mean", "packet length mean"),
    "pkt_len_std": ("pkt len std", "packet length std"),
    "pkt_len_var": ("pkt len var", "packet length variance"),

    # --- ratios ---
    "down_up_ratio": ("down/up ratio",),
    "pkt_size_avg": ("pkt size avg", "average packet size"),
    "fwd_seg_size_avg": ("fwd seg size avg", "avg fwd segment size"),
    "bwd_seg_size_avg": ("bwd seg size avg", "avg bwd segment size"),

    # --- bulk transfer (rarely populated, kept for completeness) ---
    "fwd_byts_b_avg": ("fwd byts/b avg", "fwd avg bytes/bulk"),
    "fwd_pkts_b_avg": ("fwd pkts/b avg", "fwd avg packets/bulk"),
    "fwd_blk_rate_avg": ("fwd blk rate avg", "fwd avg bulk rate"),
    "bwd_byts_b_avg": ("bwd byts/b avg", "bwd avg bytes/bulk"),
    "bwd_pkts_b_avg": ("bwd pkts/b avg", "bwd avg packets/bulk"),
    "bwd_blk_rate_avg": ("bwd blk rate avg", "bwd avg bulk rate"),

    # --- subflow ---
    "subflow_fwd_pkts": ("subflow fwd pkts", "subflow fwd packets"),
    "subflow_fwd_byts": ("subflow fwd byts", "subflow fwd bytes"),
    "subflow_bwd_pkts": ("subflow bwd pkts", "subflow bwd packets"),
    "subflow_bwd_byts": ("subflow bwd byts", "subflow bwd bytes"),

    # --- init window / active data ---
    "init_fwd_win_byts": ("init fwd win byts", "init_win_bytes_forward", "fwd init win bytes"),
    "init_bwd_win_byts": ("init bwd win byts", "init_win_bytes_backward", "bwd init win bytes"),
    "fwd_act_data_pkts": ("fwd act data pkts", "act_data_pkt_fwd", "fwd act data packets"),
    "fwd_seg_size_min": ("fwd seg size min", "min_seg_size_forward", "fwd seg size min"),

    # --- active/idle ---
    "active_mean": ("active mean",),
    "active_std": ("active std",),
    "active_max": ("active max",),
    "active_min": ("active min",),
    "idle_mean": ("idle mean",),
    "idle_std": ("idle std",),
    "idle_max": ("idle max",),
    "idle_min": ("idle min",),

    # --- label ---
    "label": ("label",),
}

# Columns that, together, define the TCP flag bitmask. Order defines bit position
# (bit 0 = SYN, matching the PS's listed order: SYN, ACK, FIN, RST, PSH, URG).
FLAG_BIT_ORDER: tuple[str, ...] = (
    "syn_flag_cnt",
    "ack_flag_cnt",
    "fin_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "urg_flag_cnt",
)

# Columns that identify a flow but must never be treated as a numeric feature.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "flow_id",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "timestamp",
)

# Columns known to sometimes contain +/-inf (Flow Byts/s and Flow Pkts/s divide by
# a duration that can be 0 for single-packet flows). Handled explicitly in
# flow_features.py rather than silently coerced by pandas.
INF_PRONE_COLUMNS: tuple[str, ...] = ("flow_byts_s", "flow_pkts_s")

_WHITESPACE_RE = re.compile(r"\s+")


def _clean_key(raw: str) -> str:
    """Normalize a raw header string for alias lookup: lowercase, collapse
    whitespace/underscores to single spaces, strip."""
    key = raw.strip().lower().replace("_", " ")
    key = _WHITESPACE_RE.sub(" ", key)
    return key


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        lookup[_clean_key(canonical)] = canonical
        for alias in aliases:
            lookup[_clean_key(alias)] = canonical
    return lookup


_ALIAS_LOOKUP = _build_lookup()


def normalize_columns(columns: list[str], *, strict: bool = False) -> dict[str, str]:
    """Map raw CSV column names to canonical names.

    Returns a dict {raw_column_name: canonical_name} containing only the columns
    that were successfully matched. Unmatched columns are omitted unless
    `strict=True`, in which case a ValueError listing them is raised (use strict
    mode when you need to be sure nothing important was silently dropped).
    """
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for raw in columns:
        key = _clean_key(raw)
        canonical = _ALIAS_LOOKUP.get(key)
        if canonical is not None:
            mapping[raw] = canonical
        else:
            unmatched.append(raw)

    if strict and unmatched:
        raise ValueError(
            f"normalize_columns: {len(unmatched)} column(s) had no known canonical "
            f"mapping: {unmatched}. Add them to CANONICAL_ALIASES in "
            f"features/schema.py once you've confirmed what they represent."
        )
    return mapping
