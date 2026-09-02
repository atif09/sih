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
                    "explanation", "warnings"):
            assert key in body, f"missing key {key}"
        if body["timeline"]:
            entry = body["timeline"][0]
            assert set(entry) == {"window_id", "window_start", "infiltration_probability", "mitre_stage"}
        stage = body["forecast"][0]["mitre_stage"]
        assert set(stage) == {"stage", "confidence", "basis", "predicted_attack_type"}

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
