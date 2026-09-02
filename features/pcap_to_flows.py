"""
PCAP -> flow-level CSV, via the `cicflowmeter` Python port (PyPI package),
the same tool family that generated CIC-IDS2017/2018's training data in the
first place -- the most correct choice for turning a standalone PCAP upload
into the flow-level feature schema features/flow_features.py already
expects, rather than reimplementing CICFlowMeter's flow-assembly logic from
scratch.

Column-name luck, verified rather than assumed: this port's CSV output
columns (src_ip, tot_fwd_pkts, flow_byts_s, syn_flag_cnt, ...) already use
the same snake_case naming as our canonical schema (features/schema.py) --
confirmed empirically against a synthetic capture, 82/82 columns matched
with zero unrecognized columns. No new aliases were needed.

Working around a real upstream bug (cicflowmeter 0.5.0)
-----------------------------------------------------------
The package's own CLI (`cicflowmeter` console script / `sniffer.main()`)
calls `create_sniffer(...)` with POSITIONAL arguments that don't match that
function's actual parameter order -- `args.fields` lands in the
`input_directory` slot and `args.verbose` (a bool) lands in the `fields`
slot, which then crashes on `fields.split(",")` for every single invocation
of the CLI, regardless of flags passed. Confirmed via a plain
`cicflowmeter -f <pcap> -c <out>` reproducing the crash and reading the
library's own source to find the mismatched call.

Worked around by calling `cicflowmeter.sniffer.create_sniffer()` directly
with correct keyword arguments, bypassing the broken CLI entirely.

A second, undocumented behavior (not a bug, just easy to miss): the
library's `AsyncSniffer` is constructed without scapy's `session=` kwarg
(only `prn=session.process`), so the session's `toPacketList()` finalization
hook -- which is what flushes any still-open flows to the output file --
never fires automatically when the offline capture ends. Confirmed
empirically: without an explicit final flush, the output CSV was created
but left completely empty (0 rows) even though flows were clearly
processed. Fixed by calling `session.garbage_collect(None)` (what
`toPacketList()` itself calls) immediately after `sniffer.join()`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from features.flow_features import extract_flow_features, load_cicflowmeter_csv


def pcap_to_flow_csv(pcap_path: str | Path, output_csv_path: str | Path | None = None) -> Path:
    """Runs the pcap through cicflowmeter and returns the path to the
    resulting flow-level CSV. If output_csv_path is None, writes to a
    temp file (caller is responsible for cleaning it up)."""
    from cicflowmeter.sniffer import create_sniffer  # imported lazily: optional dependency, only needed for PCAP-only uploads

    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"{pcap_path} does not exist")

    if output_csv_path is None:
        fd, tmp_name = tempfile.mkstemp(suffix="_flows.csv")
        import os
        os.close(fd)
        output_csv_path = Path(tmp_name)
    else:
        output_csv_path = Path(output_csv_path)

    sniffer, session = create_sniffer(
        input_file=str(pcap_path), input_interface=None, output_mode="csv",
        output=str(output_csv_path), input_directory=None, fields=None, verbose=False,
    )
    sniffer.start()
    sniffer.join()
    # See module docstring: this final flush does not happen automatically.
    session.garbage_collect(None)

    return output_csv_path


def extract_flow_features_from_pcap(pcap_path: str | Path) -> pd.DataFrame:
    """Convenience wrapper: PCAP -> flow CSV (cicflowmeter) -> canonical
    flow-level feature DataFrame (features.flow_features), in one call.
    The intermediate CSV is written to a temp file and removed afterward."""
    csv_path = pcap_to_flow_csv(pcap_path)
    try:
        raw = load_cicflowmeter_csv(csv_path)
        if raw.empty:
            raise ValueError(
                f"cicflowmeter produced no flows from {pcap_path} -- the "
                f"capture may be empty, non-IP, or too short to form a "
                f"complete flow."
            )
        return extract_flow_features(raw)
    finally:
        csv_path.unlink(missing_ok=True)
