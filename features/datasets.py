"""
Dataset registry: canonical file locations and known per-file quirks for each
supported CICFlowMeter-derived dataset.

Centralizing this here keeps flow_features.py / windowing.py fully
dataset-agnostic -- they only ever see a path and a DataFrame. Path-finding,
per-file metadata (attack categories, day), and dataset-specific gotchas
(encoding, folder-naming quirks) live in one place instead of being
rediscovered by scripts.

DATASET STATUS (as instructed by the user): CIC-IDS2017 "GeneratedLabelledFlows"
is the finalized primary dataset. CIC-IDS2018 support (features/schema.py's
alias table was built against both) is kept since it cost nothing to support
and the files are already downloaded, but it is no longer the primary target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# CIC-IDS2017 "GeneratedLabelledFlows" ships its 8 per-day/per-attack CSVs
# inside a subfolder literally named "TrafficLabelling " -- WITH a trailing
# space in the folder name. This is not a typo; it's how the dataset authors
# named it, and it's confirmed present in the actual downloaded copy. Keep it
# as a single Path segment (as below) rather than building the path via
# string concatenation/formatting that might trim trailing whitespace.
CICIDS2017_ROOT = Path("data/cicids2017")
CICIDS2017_SUBDIR = "TrafficLabelling "  # trailing space is intentional
CICIDS2017_DIR = CICIDS2017_ROOT / CICIDS2017_SUBDIR


@dataclass(frozen=True)
class CICIDS2017File:
    filename: str
    day: str
    # Attack categories as documented by the dataset authors for this
    # capture window; the Label column (per-flow) is the actual ground truth
    # used for anything training/eval related -- this field is metadata for
    # humans/hold-out-split bookkeeping, not something to train against.
    attack_categories: tuple[str, ...]
    # True if the raw file contains non-UTF-8 bytes (observed empirically:
    # the Thursday web-attacks file has a byte 0x96 -- a Windows-1252/Latin-1
    # dash character -- embedded in the data, likely from a payload string
    # captured verbatim). load_cicflowmeter_csv() falls back to latin-1
    # automatically for any file that fails a strict UTF-8 read, so this flag
    # is informational (documents *why*, for whoever reads this next) rather
    # than something calling code needs to branch on.
    has_encoding_quirk: bool = False


# Order matches the capture week. Kept as a tuple (not a dict) because the
# "day" is meaningful sequence information, not just a lookup key.
CICIDS2017_FILES: tuple[CICIDS2017File, ...] = (
    CICIDS2017File("Monday-WorkingHours.pcap_ISCX.csv", "Monday", ("Benign",)),
    CICIDS2017File("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday", ("FTP-Patator", "SSH-Patator")),
    CICIDS2017File(
        "Wednesday-workingHours.pcap_ISCX.csv", "Wednesday",
        ("DoS GoldenEye", "DoS Slowloris", "DoS Slowhttptest", "DoS Hulk", "Heartbleed"),
    ),
    CICIDS2017File(
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Thursday-Morning",
        ("Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - SQL Injection"),
        has_encoding_quirk=True,
    ),
    CICIDS2017File(
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Thursday-Afternoon",
        ("Infiltration",),
    ),
    CICIDS2017File("Friday-WorkingHours-Morning.pcap_ISCX.csv", "Friday-Morning", ("Benign",)),
    CICIDS2017File(
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday-Afternoon-PortScan",
        ("PortScan",),
    ),
    CICIDS2017File(
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Friday-Afternoon-DDoS",
        ("DDoS",),
    ),
)


def cicids2017_path(file: CICIDS2017File, base_dir: Path | str = CICIDS2017_DIR) -> Path:
    return Path(base_dir) / file.filename


def list_cicids2017_files(base_dir: Path | str = CICIDS2017_DIR) -> list[Path]:
    """Return the 8 CIC-IDS2017 day/attack-type CSV paths, in capture-week
    order. Raises FileNotFoundError (naming every missing file) if the
    directory or any expected file is absent, rather than silently returning
    a partial list."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"{base_dir} does not exist. Expected the CIC-IDS2017 "
            f"'GeneratedLabelledFlows' TrafficLabelling folder (note: the "
            f"real folder name has a trailing space) at this path."
        )
    paths = [cicids2017_path(f, base_dir) for f in CICIDS2017_FILES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{base_dir}: missing expected file(s): {missing}")
    return paths
