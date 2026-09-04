"""
Integration tests for the FastAPI app (Phase 7), using Starlette's
TestClient (in-process, no real network/server needed). Requires a trained
Phase 3 checkpoint at checkpoints/phase3/ (run models/train.py first) --
skipped automatically if it's not present, e.g. in an environment that has
never run training.
"""

from pathlib import Path

import pytest

scapy = pytest.importorskip("scapy.all")

CHECKPOINT_DIR = Path("checkpoints/phase3")
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "best_model.pt").exists(),
    reason="No trained checkpoint at checkpoints/phase3/ -- run models/train.py first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:  # triggers lifespan startup (loads the model once for all tests)
        yield c


@pytest.fixture(scope="module")
def sample_csv_bytes():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.generate_synthetic_fixture import build_fixture
    df = build_fixture(n_benign=300)
    return df.to_csv(index=False).encode()


@pytest.fixture(scope="module")
def sample_pcap_bytes(tmp_path_factory):
    # Spans >= 2 full 60s windows (the pipeline's default window_size_seconds)
    # -- a shorter capture legitimately produces < 2 windows and correctly
    # 4xxs (see test_pcap_produces_client_error_when_too_short below), which
    # isn't what this fixture is for.
    tmp_dir = tmp_path_factory.mktemp("pcap")
    pcap_path = tmp_dir / "sample.pcap"
    packets = []
    t = 0.0
    for minute in range(3):
        base = minute * 60.0
        for i in range(30):
            p = (
                scapy.IP(src="192.168.1.10", dst="10.0.0.5", ttl=64)
                / scapy.TCP(sport=44000, dport=443, seq=i * 100, window=8192, flags="PA")
                / scapy.Raw(load=b"x" * 100)
            )
            p.time = base + i * 0.5
            packets.append(p)
        for i, port in enumerate(range(2000, 2020)):
            p = scapy.IP(src="172.16.0.99", dst="10.0.0.5", ttl=48) / scapy.TCP(
                sport=50000 + i, dport=port, seq=1, window=1024, flags="S"
            )
            p.time = base + 30.0 + i
            packets.append(p)
    scapy.wrpcap(str(pcap_path), packets)
    return pcap_path.read_bytes()


class TestHealth:
    def test_health_ok_after_startup(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["window_size_seconds"] is not None

    def test_model_info(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        body = resp.json()
        assert "temporal_type" in body
        assert body["n_parameters"] > 0


class TestAnalyze:
    def test_analyze_requires_at_least_one_file(self, client):
        resp = client.post("/analyze")
        assert resp.status_code == 422

    def test_analyze_csv_only(self, client, sample_csv_bytes):
        resp = client.post(
            "/analyze",
            files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["file_type"] == "csv"
        assert body["num_windows"] > 0
        assert len(body["forecast"]) == 10  # default k
        assert "summary" in body["explanation"]
        assert 0.0 <= body["forecast"][0]["infiltration_probability"] <= 1.0
        assert any("packet-level features" in w for w in body["warnings"])

    def test_analyze_pcap_only(self, client, sample_pcap_bytes):
        resp = client.post(
            "/analyze",
            files={"pcap_file": ("sample.pcap", sample_pcap_bytes, "application/octet-stream")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["file_type"] == "pcap"
        assert any("cicflowmeter" in w for w in body["warnings"])

    def test_analyze_csv_and_pcap(self, client, sample_csv_bytes, sample_pcap_bytes):
        resp = client.post(
            "/analyze",
            files={
                "csv_file": ("sample.csv", sample_csv_bytes, "text/csv"),
                "pcap_file": ("sample.pcap", sample_pcap_bytes, "application/octet-stream"),
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["file_type"] == "csv+pcap"

    def test_analyze_respects_k_param(self, client, sample_csv_bytes):
        resp = client.post(
            "/analyze",
            files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")},
            params={"k": 3},
        )
        assert resp.status_code == 200
        assert len(resp.json()["forecast"]) == 3

    def test_analyze_response_schema_shape(self, client, sample_csv_bytes):
        resp = client.post(
            "/analyze",
            files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")},
        )
        body = resp.json()
        for key in ("filename", "file_type", "window_size_seconds", "num_windows",
                    "context_len", "timeline", "forecast", "flagged_flows",
                    "explanation", "momentum", "containment", "multiple_futures",
                    "attack_fingerprint", "warnings"):
            assert key in body, f"missing key {key}"
        if body["timeline"]:
            entry = body["timeline"][0]
            assert set(entry) == {"window_id", "window_start", "infiltration_probability", "mitre_stage", "top_flows"}
        stage = body["forecast"][0]["mitre_stage"]
        assert set(stage) == {"stage", "confidence", "basis", "predicted_attack_type"}

    def test_analyze_timeline_carries_per_window_attention(self, client):
        # Attack Future Replay: every observed timeline window should carry
        # its OWN real attention-ranked flows, not just the final window's
        # (which is what `flagged_flows` at the top level already gives).
        # The standard sample_csv_bytes fixture spans only ~21 windows,
        # right at the model's seq_len=20 context boundary -- too few valid
        # timeline positions to reliably exercise this. Doubling it (a
        # second copy of the same traffic shifted 30 minutes later) is a
        # cheap, deterministic way to guarantee enough windows without
        # depending on real held-out data inside a unit test.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import pandas as pd
        from scripts.generate_synthetic_fixture import build_fixture
        df = build_fixture(n_benign=300)
        shifted = df.copy()
        shifted["Timestamp"] = pd.to_datetime(shifted["Timestamp"]) + pd.Timedelta(minutes=30)
        big_df = pd.concat([df, shifted], ignore_index=True)
        big_csv = big_df.to_csv(index=False).encode()

        resp = client.post("/analyze", files={"csv_file": ("sample.csv", big_csv, "text/csv")})
        body = resp.json()
        assert len(body["timeline"]) > 1
        for entry in body["timeline"]:
            assert isinstance(entry["top_flows"], list)
            for flow in entry["top_flows"]:
                assert set(flow) == {
                    "src_ip", "dst_ip", "attention_weight", "attention_concentration",
                    "flow_count", "bytes_sum", "dst_port_nunique", "retransmission_sum", "is_attack_edge",
                }
        # Different windows should not all carry identical top_flows -- real
        # per-window attention, not one value copy-pasted across entries.
        distinct_first_flows = {tuple(sorted(e["top_flows"][0].items())) for e in body["timeline"] if e["top_flows"]}
        assert len(distinct_first_flows) > 1

    def test_analyze_momentum_shape(self, client, sample_csv_bytes):
        resp = client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        momentum = resp.json()["momentum"]
        assert set(momentum) == {"score", "top_driver", "explanation", "attack_path_expansion_available"}
        assert 0.0 <= momentum["score"] <= 100.0
        assert momentum["attack_path_expansion_available"] is True  # Multiple Parallel Futures (Phase 10) now feeds this

    def test_analyze_containment_shape(self, client, sample_csv_bytes):
        resp = client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        containment = resp.json()["containment"]
        assert set(containment) == {
            "current_stage", "likely_range_minutes", "full_observed_range_minutes",
            "n_historical_episodes", "has_sufficient_data", "recommended_action", "rationale",
        }
        # Whichever stage this fixture lands on, a range is present exactly
        # when there's historical data for it -- never a dangling range with
        # zero backing episodes.
        if containment["n_historical_episodes"] == 0:
            assert containment["likely_range_minutes"] is None
        else:
            assert containment["likely_range_minutes"] is not None
            assert len(containment["likely_range_minutes"]) == 2

    def test_analyze_multiple_futures_shape(self, client, sample_csv_bytes):
        resp = client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        futures = resp.json()["multiple_futures"]
        assert set(futures) == {"n_samples", "outcomes", "deterministic_stage"}
        assert futures["n_samples"] > 0
        assert len(futures["outcomes"]) >= 1
        probs = [o["probability"] for o in futures["outcomes"]]
        assert probs == sorted(probs, reverse=True)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        for outcome in futures["outcomes"]:
            assert set(outcome) == {"stage", "probability", "n_samples", "mean_infiltration_probability"}
            assert 0.0 <= outcome["mean_infiltration_probability"] <= 1.0

    def test_analyze_attack_fingerprint_shape(self, client, sample_csv_bytes):
        resp = client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        fp = resp.json()["attack_fingerprint"]
        assert set(fp) == {"current", "predicted", "comparison", "top_stage", "summary"}
        assert len(fp["current"]) == 6  # all 6 real MITRE stages, always
        assert len(fp["predicted"]) == 6
        assert len(fp["comparison"]) == 6
        for entry in fp["current"]:
            assert set(entry) == {"stage", "score", "similarity", "driving_attack_type"}
            assert entry["similarity"] in ("High", "Medium", "Low")
        exfil = next(e for e in fp["current"] if e["stage"] == "Exfiltration")
        assert exfil["driving_attack_type"] is None  # no attack type maps to Exfiltration, see module docstring
        for entry in fp["comparison"]:
            assert entry["trend"] in ("rising", "falling", "stable")

    def test_pcap_too_short_returns_client_error_not_crash(self, client, tmp_path):
        pcap_path = tmp_path / "short.pcap"
        p = scapy.IP(src="10.0.0.1", dst="10.0.0.2", ttl=64) / scapy.TCP(sport=1000, dport=80, seq=1, window=8192, flags="S")
        p.time = 0.0
        scapy.wrpcap(str(pcap_path), [p])
        resp = client.post(
            "/analyze",
            files={"pcap_file": ("short.pcap", pcap_path.read_bytes(), "application/octet-stream")},
        )
        assert resp.status_code == 422
        assert "time window" in resp.json()["detail"]

    def test_analyze_garbage_csv_returns_client_error_not_crash(self, client):
        resp = client.post(
            "/analyze",
            files={"csv_file": ("garbage.csv", b"not,a,valid\nflow,csv,file\n", "text/csv")},
        )
        assert resp.status_code in (422, 500)
        assert "detail" in resp.json()


class TestSimulate:
    """Phase 9, Feature 3 endpoint wiring. /simulate acts on the context
    cached from the most recently completed /analyze call in this process,
    so every test here calls /analyze first."""

    def test_simulate_before_any_analyze_returns_client_error(self, client):
        # By this point in the module other tests have already populated the
        # cached context via /analyze (the `client` fixture is module-scoped)
        # -- can't spin up a second TestClient to get a genuinely fresh
        # _state, since exiting it would run lifespan shutdown and clear the
        # SAME global _state dict out from under the still-open module-scoped
        # client. Instead, directly clear just the cached-context key to
        # reproduce the "no /analyze yet" state, then restore it afterwards.
        from api.main import _state
        saved = _state.pop("last_context", None)
        try:
            resp = client.post("/simulate", json={"action": "isolate_host", "host_ip": "172.16.0.99"})
            assert resp.status_code == 422
            assert "analyze" in resp.json()["detail"]
        finally:
            if saved is not None:
                _state["last_context"] = saved

    def test_simulate_isolate_host_after_analyze(self, client, sample_csv_bytes):
        analyze_resp = client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        assert analyze_resp.status_code == 200

        resp = client.post("/simulate", json={"action": "isolate_host", "host_ip": "172.16.0.99", "k": 3})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["original_probability"]) == 3
        assert len(body["modified_probability"]) == 3
        assert len(body["probability_delta"]) == 3
        assert body["probability_delta"] == pytest.approx(
            [m - o for m, o in zip(body["modified_probability"], body["original_probability"])]
        )
        assert "172.16.0.99" in body["action_label"]
        assert isinstance(body["side_effects"], list)

    def test_simulate_block_flow_after_analyze(self, client, sample_csv_bytes):
        analyze_resp = client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        assert analyze_resp.status_code == 200

        resp = client.post("/simulate", json={"action": "block_flow", "src_ip": "172.16.0.99", "dst_ip": "10.0.0.5"})
        assert resp.status_code == 200, resp.text
        assert "10.0.0.5" in resp.json()["action_label"]

    def test_simulate_unknown_action_is_client_error(self, client, sample_csv_bytes):
        client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        resp = client.post("/simulate", json={"action": "nuke_everything"})
        assert resp.status_code == 422

    def test_simulate_isolate_host_missing_host_ip_is_client_error(self, client, sample_csv_bytes):
        client.post("/analyze", files={"csv_file": ("sample.csv", sample_csv_bytes, "text/csv")})
        resp = client.post("/simulate", json={"action": "isolate_host"})
        assert resp.status_code == 422
