"""
Generate a small SYNTHETIC CICFlowMeter-shaped CSV for sanity-checking the
Phase 1 pipeline before real CIC-IDS2018 data is downloaded.

This is NOT real traffic and must never be used for anything beyond smoke-
testing the extractor code (column handling, dtype coercion, windowing). It
mimics: a burst of benign traffic, a sequential port scan (Reconnaissance-
shaped), and a DoS-shaped burst (many packets, tiny inter-arrival time) --
enough shape diversity to exercise every branch in flow_features.py.

Usage:
    python scripts/generate_synthetic_fixture.py [output_csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

CICFLOWMETER_COLUMNS = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio",
    "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label",
]


def _row(src_ip, src_port, dst_ip, dst_port, proto, ts, label, *,
         syn=0, ack=0, fin=0, rst=0, psh=0, urg=0,
         fwd_pkts=1, bwd_pkts=1, duration=1000, iat_mean=100.0):
    fwd_bytes = int(RNG.integers(40, 1500) * fwd_pkts)
    bwd_bytes = int(RNG.integers(40, 1500) * bwd_pkts)
    row = {c: 0 for c in CICFLOWMETER_COLUMNS}
    row.update({
        "Flow ID": f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{proto}",
        "Src IP": src_ip, "Src Port": src_port, "Dst IP": dst_ip, "Dst Port": dst_port,
        "Protocol": proto, "Timestamp": ts,
        "Flow Duration": duration,
        "Tot Fwd Pkts": fwd_pkts, "Tot Bwd Pkts": bwd_pkts,
        "TotLen Fwd Pkts": fwd_bytes, "TotLen Bwd Pkts": bwd_bytes,
        "Flow Byts/s": (fwd_bytes + bwd_bytes) / max(duration, 1) * 1e6,
        "Flow Pkts/s": (fwd_pkts + bwd_pkts) / max(duration, 1) * 1e6,
        "Flow IAT Mean": iat_mean, "Flow IAT Std": iat_mean * 0.1, "Flow IAT Max": iat_mean * 2,
        "SYN Flag Cnt": syn, "ACK Flag Cnt": ack, "FIN Flag Cnt": fin,
        "RST Flag Cnt": rst, "PSH Flag Cnt": psh, "URG Flag Cnt": urg,
        "Down/Up Ratio": bwd_pkts / max(fwd_pkts, 1),
        "Label": label,
    })
    return row


def build_fixture(n_benign: int = 200) -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2018-03-02 09:00:00")

    # 1. Benign background traffic: normal-looking HTTP/HTTPS-ish flows,
    # spread over 10 minutes, random destination ports from a small set.
    for i in range(n_benign):
        ts = base_ts + pd.Timedelta(seconds=int(RNG.integers(0, 600)))
        rows.append(_row(
            f"192.168.1.{RNG.integers(2, 50)}", int(RNG.integers(1024, 65535)),
            f"10.0.0.{RNG.integers(2, 20)}", int(RNG.choice([80, 443, 53])),
            6, ts, "Benign",
            syn=1, ack=1, fin=1, fwd_pkts=int(RNG.integers(2, 20)),
            bwd_pkts=int(RNG.integers(2, 20)), duration=int(RNG.integers(1000, 500000)),
            iat_mean=float(RNG.uniform(500, 5000)),
        ))

    # 2. Sequential port scan (Reconnaissance-shaped): one attacker IP hits
    # 30 consecutive destination ports on one target, ~1 packet each,
    # within a single ~30-second window.
    attacker_ip = "172.16.0.99"
    scan_start = base_ts + pd.Timedelta(minutes=15)
    for i, port in enumerate(range(1000, 1030)):
        ts = scan_start + pd.Timedelta(seconds=i)
        rows.append(_row(
            attacker_ip, int(RNG.integers(40000, 60000)), "10.0.0.5", port, 6, ts,
            "PortScan", syn=1, rst=1, fwd_pkts=1, bwd_pkts=1,
            duration=int(RNG.integers(100, 5000)), iat_mean=50.0,
        ))

    # 3. DoS-shaped burst: many small flows in a tight time window, low IAT,
    # SYN-heavy, mostly one-directional (few/no backward packets).
    dos_start = base_ts + pd.Timedelta(minutes=20)
    for i in range(150):
        ts = dos_start + pd.Timedelta(milliseconds=int(RNG.integers(0, 2000)))
        rows.append(_row(
            f"172.16.0.{RNG.integers(100, 200)}", int(RNG.integers(1024, 65535)),
            "10.0.0.5", 80, 6, ts, "DoS",
            syn=1, fwd_pkts=1, bwd_pkts=0, duration=int(RNG.integers(1, 100)),
            iat_mean=5.0,
        ))

    df = pd.DataFrame(rows, columns=CICFLOWMETER_COLUMNS)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cicids2018/_synthetic_sample.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_fixture()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out_path}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
