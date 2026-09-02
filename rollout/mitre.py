"""
MITRE ATT&CK stage mapping: turns a predicted future network state into one
of the PS's five named stages (Reconnaissance, Initial Access, Lateral
Movement, Command & Control, Exfiltration).

This mapping is explicitly a JUDGMENT CALL -- the PS leaves the exact logic
open ("Document your mapping logic clearly in code comments since this is a
judgment call the PS leaves open to us"). Documented in full below rather
than left implicit.

Two-tier design
----------------
1. **Primary: predicted attack-type -> stage**, via `ATTACK_TYPE_TO_STAGE`.
   The world model's auxiliary attack-type head (models/world_model.py)
   predicts a probability per CIC-IDS2017 attack category (the categories
   observed in the TRAINING days only -- see models/dataset.py). Each
   category is mapped to the kill-chain stage it best represents. This is
   the primary path because it's grounded in the model's own supervised
   signal, not a hand-tuned threshold on raw features.

2. **Fallback: raw-feature heuristic**, used when either (a) the attack-type
   head's top prediction is below `TYPE_CONFIDENCE_THRESHOLD` (the model
   isn't confident enough to trust a specific category -- notably including
   every prediction on an unseen/held-out attack type, which structurally
   cannot appear in the attack-type vocabulary), or (b) the infiltration
   probability itself is high but no category is confident at all. This
   fallback looks at the *last real observed context snapshot's* aggregate
   node/edge statistics (port-scan score, port diversity, outbound byte
   asymmetry) -- it cannot look at predicted future windows' raw features,
   because the world model only predicts a latent embedding for the future,
   not decoded raw features (see world_model.py's docstring on why).

Known limitation, stated plainly: CIC-IDS2017 has no dataset category that
naturally maps to "Exfiltration" or "Command & Control" as cleanly as
PortScan maps to "Reconnaissance" -- Bot (botnet/ARES C2 beaconing) is the
closest fit for Command & Control, and Exfiltration is reachable only through
the raw-feature fallback heuristic (sustained high outbound-byte asymmetry),
never through the attack-type path. This is disclosed rather than papered
over with a falsely-confident mapping.

Attack types that don't fit any of the PS's five stages (the DoS/DDoS
family, and Heartbleed) are mapped to an explicit sixth label,
"Denial of Service / Impact", rather than force-fit into one of the five.
Misclassifying a volumetric flood as, say, "Lateral Movement" would be a
worse and more misleading judgment call than admitting the PS's five-stage
taxonomy doesn't cover every attack category in this dataset. Consumers that
strictly need one of the five PS stages can treat this label as "not
applicable" / filter it out; it is kept distinct rather than silently folded
into the closest of the five so the honesty of the mapping isn't lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph.snapshot import GraphSnapshot, NODE_FEATURE_NAMES

STAGE_RECONNAISSANCE = "Reconnaissance"
STAGE_INITIAL_ACCESS = "Initial Access"
STAGE_LATERAL_MOVEMENT = "Lateral Movement"
STAGE_COMMAND_AND_CONTROL = "Command and Control"
STAGE_EXFILTRATION = "Exfiltration"
STAGE_DOS_IMPACT = "Denial of Service / Impact"  # explicit 6th label, see module docstring
STAGE_BENIGN = "Benign / No Stage"

# Primary mapping: CIC-IDS2017 attack-type label -> best-fit MITRE stage.
#
#   PortScan                    -> Reconnaissance: literally port/host discovery (T1046, T1595).
#   FTP-Patator, SSH-Patator    -> Initial Access: credential brute-forcing to gain a foothold
#                                  (arguably "Credential Access" in full ATT&CK, but the PS's
#                                  5-stage kill-chain framing has no such bucket; "the attacker is
#                                  trying to get in" fits Initial Access best of the five).
#   Web Attack - *              -> Initial Access: exploiting a public-facing web app to gain a
#                                  foothold (T1190).
#   Infiltration                -> Lateral Movement: CIC-IDS2017's "Infiltration" scenario is a
#                                  machine already compromised via a dropped executable, then
#                                  performing internal reconnaissance/pivoting against other
#                                  internal hosts -- i.e. it documents the POST-compromise internal
#                                  movement phase, not the initial entry.
#   Bot                         -> Command and Control: botnet (ARES) traffic is C2 beaconing by
#                                  definition.
#   DoS *, Heartbleed           -> Denial of Service / Impact (explicit 6th label, not one of the
#                                  five -- see module docstring). Heartbleed here is used as a
#                                  crash/DoS-style exploit in the dataset's capture, not a stealthy
#                                  data-exfiltration read (which is Heartbleed's other real-world
#                                  use) -- documented as a judgment call.
ATTACK_TYPE_TO_STAGE: dict[str, str] = {
    "PortScan": STAGE_RECONNAISSANCE,
    "FTP-Patator": STAGE_INITIAL_ACCESS,
    "FTP - Patator": STAGE_INITIAL_ACCESS,  # label-text variant seen in the real data
    "SSH-Patator": STAGE_INITIAL_ACCESS,
    "SSH - Patator": STAGE_INITIAL_ACCESS,
    "Web Attack - Brute Force": STAGE_INITIAL_ACCESS,
    "Web Attack - XSS": STAGE_INITIAL_ACCESS,
    "Web Attack - Sql Injection": STAGE_INITIAL_ACCESS,
    "Infiltration": STAGE_LATERAL_MOVEMENT,
    "Bot": STAGE_COMMAND_AND_CONTROL,
    "DoS GoldenEye": STAGE_DOS_IMPACT,
    "DoS Hulk": STAGE_DOS_IMPACT,
    "DoS Slowhttptest": STAGE_DOS_IMPACT,
    "DoS slowloris": STAGE_DOS_IMPACT,
    "DDoS": STAGE_DOS_IMPACT,
    "DDOS attack-HOIC": STAGE_DOS_IMPACT,
    "DDOS attack-LOIC-UDP": STAGE_DOS_IMPACT,
    "Heartbleed": STAGE_DOS_IMPACT,
}

TYPE_CONFIDENCE_THRESHOLD = 0.5

# Raw-feature fallback thresholds (used on the last REAL context snapshot's
# aggregated per-host features -- see NODE_FEATURE_NAMES). Deliberately
# simple and documented rather than learned: this is a stopgap for when the
# attack-type head isn't confident (including, structurally, every unseen
# attack type), not a replacement for the model's own predictions.
_PORT_SCAN_SCORE_IDX = NODE_FEATURE_NAMES.index("out_port_scan_score_max")
_OUT_UNIQUE_PORTS_IDX = NODE_FEATURE_NAMES.index("out_unique_ports")
_OUT_BYTES_SUM_IDX = NODE_FEATURE_NAMES.index("out_bytes_sum")
_IN_BYTES_SUM_IDX = NODE_FEATURE_NAMES.index("in_bytes_sum")

RECON_PORT_SCAN_SCORE_THRESHOLD = 0.3
RECON_UNIQUE_PORTS_THRESHOLD = 10
EXFIL_OUTBOUND_BYTE_RATIO_THRESHOLD = 5.0  # out_bytes >> in_bytes for the most active host


@dataclass
class StagePrediction:
    stage: str
    confidence: float
    basis: str  # "attack_type" or "heuristic_fallback" or "benign" -- how this stage was decided
    predicted_attack_type: str | None = None


def _raw_feature_fallback(context_snapshot: GraphSnapshot) -> StagePrediction:
    """Heuristic stage guess from the last real observed snapshot's node
    features, used when the attack-type head isn't confident. See module
    docstring for the full rationale and known limitations."""
    if context_snapshot.num_nodes == 0:
        return StagePrediction(STAGE_BENIGN, confidence=1.0, basis="heuristic_fallback")

    feats = context_snapshot.node_features
    max_port_scan_score = float(feats[:, _PORT_SCAN_SCORE_IDX].max())
    max_unique_ports = float(feats[:, _OUT_UNIQUE_PORTS_IDX].max())

    if max_port_scan_score >= RECON_PORT_SCAN_SCORE_THRESHOLD or max_unique_ports >= RECON_UNIQUE_PORTS_THRESHOLD:
        return StagePrediction(STAGE_RECONNAISSANCE, confidence=min(max_port_scan_score, 1.0), basis="heuristic_fallback")

    out_bytes = feats[:, _OUT_BYTES_SUM_IDX]
    in_bytes = feats[:, _IN_BYTES_SUM_IDX]
    # Ratio for whichever host moved the most total traffic this window --
    # a large, mostly-outbound transfer is the simplest exfiltration proxy
    # available without decoded raw features for the future.
    busiest = int(np.argmax(out_bytes + in_bytes))
    ratio = out_bytes[busiest] / max(in_bytes[busiest], 1.0)
    if ratio >= EXFIL_OUTBOUND_BYTE_RATIO_THRESHOLD:
        return StagePrediction(STAGE_EXFILTRATION, confidence=min(ratio / (2 * EXFIL_OUTBOUND_BYTE_RATIO_THRESHOLD), 1.0), basis="heuristic_fallback")

    return StagePrediction(STAGE_INITIAL_ACCESS, confidence=0.0, basis="heuristic_fallback")


def predict_stage(
    infiltration_probability: float,
    attack_type_probs: dict[str, float] | None,
    context_snapshot: GraphSnapshot,
    infiltration_threshold: float = 0.5,
) -> StagePrediction:
    """Map one rollout step's predictions to a MITRE stage.

    infiltration_probability: sigmoid(attack_logit) for this step.
    attack_type_probs: {attack_type_name: probability} from the auxiliary
        head, or None if the model has no attack-type head.
    context_snapshot: the last REAL observed GraphSnapshot feeding the
        rollout -- used only by the raw-feature fallback (predicted future
        windows have no decoded raw features to inspect, see module
        docstring).
    """
    if infiltration_probability < infiltration_threshold:
        return StagePrediction(STAGE_BENIGN, confidence=1.0 - infiltration_probability, basis="benign")

    if attack_type_probs:
        top_type, top_prob = max(attack_type_probs.items(), key=lambda kv: kv[1])
        if top_prob >= TYPE_CONFIDENCE_THRESHOLD and top_type in ATTACK_TYPE_TO_STAGE:
            return StagePrediction(
                ATTACK_TYPE_TO_STAGE[top_type], confidence=top_prob,
                basis="attack_type", predicted_attack_type=top_type,
            )

    return _raw_feature_fallback(context_snapshot)
