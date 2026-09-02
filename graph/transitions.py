"""
Ground-truth state transition extraction: turns a day's ordered list of
GraphSnapshots into (S_t, S_t+1, attack_label_t+1) sequences for supervised
dynamics learning (Phase 3's P(S_t+1 | S_t)).

Design notes
------------
- Transitions are only ever formed **within one day/file**. Different
  CIC-IDS2017 days are independent captures (even though the private IP
  address space, e.g. 192.168.10.x, is reused across days) -- pairing the
  last window of one day with the first window of the next would fabricate a
  transition across a multi-hour real-world gap that never happened.
- Snapshots from graph.snapshot.build_snapshots_for_day are already
  gap-filled (every integer window_id present, including empty windows), so
  consecutive-index pairing is equivalent to consecutive-time pairing here.
  The strict `window_id_t1 == window_id_t + 1` check below is a defensive
  invariant, not the primary mechanism.
- `transition_type` buckets the PS's "label whether/how the state is
  progressing toward an attack" requirement into four categories from
  (attack_present_t, attack_present_t+1). `benign_to_attack` is the specific
  "progression" moment a forecasting system most wants to catch *before* it
  happens (i.e. predict it while still in a benign_to_benign or
  attack_to_attack run leading up to it) -- Phase 6's benchmark should look
  at recall on this category specifically, not just overall accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph.snapshot import GraphSnapshot

TRANSITION_BENIGN_TO_BENIGN = "benign_to_benign"
TRANSITION_BENIGN_TO_ATTACK = "benign_to_attack"
TRANSITION_ATTACK_TO_ATTACK = "attack_to_attack"
TRANSITION_ATTACK_TO_BENIGN = "attack_to_benign"


@dataclass
class Transition:
    day: str
    window_id_t: int
    window_id_t1: int
    state_t: GraphSnapshot
    state_t1: GraphSnapshot
    attack_label_next: int  # 0/1 -- graph_attack_present of state_t1, the primary training target
    attack_types_next: frozenset
    transition_type: str


def _transition_type(attack_t: bool, attack_t1: bool) -> str:
    if not attack_t and not attack_t1:
        return TRANSITION_BENIGN_TO_BENIGN
    if not attack_t and attack_t1:
        return TRANSITION_BENIGN_TO_ATTACK
    if attack_t and attack_t1:
        return TRANSITION_ATTACK_TO_ATTACK
    return TRANSITION_ATTACK_TO_BENIGN


def build_transitions(snapshots: list[GraphSnapshot]) -> list[Transition]:
    """snapshots must be for a single day, ordered by window_id ascending
    (as returned by graph.snapshot.build_snapshots_for_day)."""
    transitions: list[Transition] = []
    for s_t, s_t1 in zip(snapshots, snapshots[1:]):
        if s_t1.window_id != s_t.window_id + 1:
            # Should not happen given gap-filled input from
            # build_snapshots_for_day, but skip rather than fabricate a
            # transition across an unexpected gap if it ever does.
            continue
        if s_t.day != s_t1.day:
            continue

        transitions.append(Transition(
            day=s_t.day,
            window_id_t=s_t.window_id, window_id_t1=s_t1.window_id,
            state_t=s_t, state_t1=s_t1,
            attack_label_next=int(s_t1.graph_attack_present),
            attack_types_next=s_t1.graph_attack_types,
            transition_type=_transition_type(s_t.graph_attack_present, s_t1.graph_attack_present),
        ))
    return transitions
