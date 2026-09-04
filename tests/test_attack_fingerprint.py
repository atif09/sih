import pytest

from fingerprint.attack_fingerprint import compute_attack_fingerprint, STAGES
from rollout.mitre import (
    STAGE_RECONNAISSANCE, STAGE_COMMAND_AND_CONTROL, STAGE_EXFILTRATION,
    STAGE_INITIAL_ACCESS, STAGE_LATERAL_MOVEMENT, STAGE_DOS_IMPACT,
)


class TestStageFingerprint:
    def test_all_six_stages_present(self):
        result = compute_attack_fingerprint({"PortScan": 0.8}, [{"PortScan": 0.8}])
        stages_seen = {e.stage for e in result.current}
        assert stages_seen == {
            STAGE_RECONNAISSANCE, STAGE_INITIAL_ACCESS, STAGE_LATERAL_MOVEMENT,
            STAGE_COMMAND_AND_CONTROL, STAGE_EXFILTRATION, STAGE_DOS_IMPACT,
        }
        assert len(STAGES) == 6

    def test_exfiltration_always_zero_with_no_driver(self):
        # No attack type in ATTACK_TYPE_TO_STAGE maps to Exfiltration --
        # confirmed directly against rollout/mitre.py's real table.
        result = compute_attack_fingerprint({"PortScan": 0.9, "Bot": 0.9}, [{"PortScan": 0.9, "Bot": 0.9}])
        exfil = next(e for e in result.current if e.stage == STAGE_EXFILTRATION)
        assert exfil.score == 0.0
        assert exfil.driving_attack_type is None
        assert exfil.similarity == "Low"

    def test_high_probability_type_maps_to_high_similarity(self):
        result = compute_attack_fingerprint({"PortScan": 0.9}, [{"PortScan": 0.9}])
        recon = next(e for e in result.current if e.stage == STAGE_RECONNAISSANCE)
        assert recon.score == 0.9
        assert recon.similarity == "High"
        assert recon.driving_attack_type == "PortScan"

    def test_low_probability_maps_to_low_similarity(self):
        result = compute_attack_fingerprint({"PortScan": 0.05}, [{"PortScan": 0.05}])
        recon = next(e for e in result.current if e.stage == STAGE_RECONNAISSANCE)
        assert recon.similarity == "Low"

    def test_medium_band(self):
        result = compute_attack_fingerprint({"PortScan": 0.3}, [{"PortScan": 0.3}])
        recon = next(e for e in result.current if e.stage == STAGE_RECONNAISSANCE)
        assert recon.similarity == "Medium"

    def test_max_not_sum_across_multiple_types_same_stage(self):
        # FTP-Patator and SSH-Patator both map to Initial Access -- the
        # score must be the max of the two, never their sum (which would
        # exceed 1 and misrepresent independent sigmoid outputs as additive).
        result = compute_attack_fingerprint(
            {"FTP-Patator": 0.4, "SSH-Patator": 0.7}, [{"FTP-Patator": 0.4, "SSH-Patator": 0.7}],
        )
        access = next(e for e in result.current if e.stage == STAGE_INITIAL_ACCESS)
        assert access.score == 0.7
        assert access.driving_attack_type == "SSH-Patator"

    def test_empty_probs_gives_all_zero_fingerprint(self):
        result = compute_attack_fingerprint({}, [{}])
        assert all(e.score == 0.0 for e in result.current)
        assert all(e.driving_attack_type is None for e in result.current)

    def test_current_sorted_descending_by_score(self):
        result = compute_attack_fingerprint({"PortScan": 0.9, "Bot": 0.3}, [{"PortScan": 0.9, "Bot": 0.3}])
        scores = [e.score for e in result.current]
        assert scores == sorted(scores, reverse=True)
        assert result.current[0].stage == STAGE_RECONNAISSANCE


class TestPredictedFingerprint:
    def test_predicted_is_final_forecast_step_not_mean(self):
        # Matches Multiple Parallel Futures' own endpoint convention
        # (futures/parallel_futures.py's deterministic_stage is likewise
        # the LAST rollout step, not an average) -- and avoids diluting an
        # already-small real signal by averaging in steps closer to "now".
        forecast = [{"PortScan": 0.2}, {"PortScan": 0.4}, {"PortScan": 0.9}]
        result = compute_attack_fingerprint({"PortScan": 0.2}, forecast)
        recon = next(e for e in result.predicted if e.stage == STAGE_RECONNAISSANCE)
        assert recon.score == pytest.approx(0.9)

    def test_empty_forecast_list_handled(self):
        result = compute_attack_fingerprint({"PortScan": 0.5}, [])
        assert all(e.score == 0.0 for e in result.predicted)


class TestComparison:
    def test_rising_trend_detected(self):
        result = compute_attack_fingerprint({"Bot": 0.1}, [{"Bot": 0.1}, {"Bot": 0.9}])
        c2c = next(e for e in result.comparison if e.stage == STAGE_COMMAND_AND_CONTROL)
        assert c2c.trend == "rising"
        assert c2c.predicted_score > c2c.current_score

    def test_falling_trend_detected(self):
        result = compute_attack_fingerprint({"Bot": 0.9}, [{"Bot": 0.9}, {"Bot": 0.1}])
        c2c = next(e for e in result.comparison if e.stage == STAGE_COMMAND_AND_CONTROL)
        assert c2c.trend == "falling"

    def test_stable_trend_within_threshold(self):
        result = compute_attack_fingerprint({"Bot": 0.5}, [{"Bot": 0.5}, {"Bot": 0.5015}])
        c2c = next(e for e in result.comparison if e.stage == STAGE_COMMAND_AND_CONTROL)
        assert c2c.trend == "stable"

    def test_small_but_real_delta_is_not_swallowed_as_stable(self):
        # The threshold was originally 0.05 -- picked without checking real
        # model output magnitudes -- which made "stable" the answer for
        # every real capture tried, since this model's real attack-type
        # deltas top out around ~0.004-0.006 (see module docstring's
        # empirical note). A delta of this real scale must now register as
        # a genuine trend, not get swallowed into "stable".
        result = compute_attack_fingerprint({"Bot": 0.0341}, [{"Bot": 0.0341}, {"Bot": 0.0295}])
        c2c = next(e for e in result.comparison if e.stage == STAGE_COMMAND_AND_CONTROL)
        assert c2c.trend == "falling"

    def test_comparison_sorted_by_absolute_delta_descending(self):
        result = compute_attack_fingerprint(
            {"PortScan": 0.1, "Bot": 0.1},
            [{"PortScan": 0.1, "Bot": 0.1}, {"PortScan": 0.15, "Bot": 0.95}],
        )
        deltas = [abs(e.predicted_score - e.current_score) for e in result.comparison]
        assert deltas == sorted(deltas, reverse=True)
        assert result.comparison[0].stage == STAGE_COMMAND_AND_CONTROL


class TestSummary:
    def test_summary_names_top_stage_and_driver(self):
        result = compute_attack_fingerprint({"PortScan": 0.9}, [{"PortScan": 0.9}])
        assert STAGE_RECONNAISSANCE in result.summary
        assert "PortScan" in result.summary

    def test_summary_handles_no_signal_case(self):
        result = compute_attack_fingerprint({}, [{}])
        assert "no strong resemblance" in result.summary
