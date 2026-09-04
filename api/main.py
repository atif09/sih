"""
FastAPI application wrapping the full inference pipeline. Fully offline --
no cloud API calls anywhere in the inference path (the trained checkpoint is
loaded from local disk once at startup; cicflowmeter/scapy/torch all run
locally).

Run:
    uvicorn api.main:app --reload --port 8000

Then see http://localhost:8000/docs for the interactive OpenAPI UI, or POST
directly:
    curl -X POST http://localhost:8000/analyze \
        -F "csv_file=@data/cicids2017/TrafficLabelling /Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.pipeline import InsufficientDataError, ModelBundle, run_pipeline, run_simulation
from api.schemas import AnalyzeResponse, ErrorResponse, ModelValidationResponse, SimulateRequest, SimulateResponse
from models.train import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = "checkpoints/phase3"
DEFAULT_PIPELINE_CONFIG = "configs/pipeline.yaml"
BENCHMARK_REPORT_PATH = Path("benchmark/report.json")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device("auto")
    logger.info(f"Loading world model from {DEFAULT_CHECKPOINT_DIR} onto {device} ...")
    _state["bundle"] = ModelBundle.load(DEFAULT_CHECKPOINT_DIR, DEFAULT_PIPELINE_CONFIG, device)
    logger.info("Model loaded.")

    # Benchmark result is a fixed, precomputed MODEL property (see
    # benchmark/run_benchmark.py), not something derived from any request --
    # loaded once at startup like the model itself, not recomputed per call.
    if BENCHMARK_REPORT_PATH.exists():
        _state["benchmark_report"] = json.loads(BENCHMARK_REPORT_PATH.read_text())
        logger.info(f"Loaded benchmark report from {BENCHMARK_REPORT_PATH}.")
    else:
        _state["benchmark_report"] = None
        logger.warning(f"{BENCHMARK_REPORT_PATH} not found -- /model-validation will 404 until `python benchmark/run_benchmark.py` is run.")

    yield
    _state.clear()


app = FastAPI(
    title="Network Attack Forecasting API",
    description="World-model-based network attack forecasting (SIH26153). "
                "Fully offline inference: feature extraction -> graph construction "
                "-> world model -> K-step rollout -> MITRE stage mapping -> explainability.",
    version="0.1.0",
    lifespan=lifespan,
)

# Phase 8's frontend (Vite dev server) calls this API directly from the
# browser. The Vite dev server also proxies /api -> here, but CORS is added
# regardless so the API works when accessed cross-origin without the proxy
# too (e.g. a built frontend served separately). Local-only origins -- this
# is an offline demo, not a public API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    bundle: ModelBundle | None = _state.get("bundle")
    return {
        "status": "ok" if bundle is not None else "model_not_loaded",
        "device": str(bundle.device) if bundle else None,
        "window_size_seconds": bundle.window_size_seconds if bundle else None,
        "seq_len": bundle.seq_len if bundle else None,
    }


@app.get("/model-info")
def model_info():
    bundle: ModelBundle = _state["bundle"]
    return {
        "temporal_type": bundle.model.config.temporal_type,
        "latent_dim": bundle.model.config.latent_dim,
        "attack_type_vocab": bundle.model.config.attack_type_vocab,
        "window_size_seconds": bundle.window_size_seconds,
        "seq_len": bundle.seq_len,
        "default_k": bundle.default_k,
        "n_parameters": sum(p.numel() for p in bundle.model.parameters()),
    }


@app.get(
    "/model-validation",
    response_model=ModelValidationResponse,
    responses={404: {"model": ErrorResponse}},
)
def model_validation():
    """Live version of benchmark/report.md's static prose -- world model vs.
    logistic-regression baseline, on the real validation split and the
    held-out (unseen attack type) day, both computed once by
    benchmark/run_benchmark.py and served here unchanged. Independent of any
    /analyze call: this describes the MODEL, not a particular upload."""
    report = _state.get("benchmark_report")
    if report is None:
        raise HTTPException(status_code=404, detail=f"No benchmark report found at {BENCHMARK_REPORT_PATH} -- run `python benchmark/run_benchmark.py` first.")
    return ModelValidationResponse(**report)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analyze(
    csv_file: UploadFile | None = File(None, description="CICFlowMeter-style flow CSV"),
    pcap_file: UploadFile | None = File(None, description="Raw packet capture"),
    k: int = 10,
):
    """Accepts a CSV, a PCAP, or both (see api/pipeline.py's module docstring
    for exactly how each combination is handled). At least one is required.
    """
    if csv_file is None and pcap_file is None:
        raise HTTPException(status_code=422, detail="Provide at least one of csv_file or pcap_file.")

    bundle: ModelBundle = _state["bundle"]
    tmp_dir = Path(tempfile.mkdtemp(prefix="analyze_"))
    csv_path = pcap_path = None
    try:
        if csv_file is not None:
            csv_path = tmp_dir / (csv_file.filename or "upload.csv")
            with open(csv_path, "wb") as f:
                shutil.copyfileobj(csv_file.file, f)
        if pcap_file is not None:
            pcap_path = tmp_dir / (pcap_file.filename or "upload.pcap")
            with open(pcap_path, "wb") as f:
                shutil.copyfileobj(pcap_file.file, f)

        result, final_context = run_pipeline(bundle, csv_path, pcap_path, k=k)
        # Cached for a later /simulate call (Phase 9, Feature 3) -- see that
        # endpoint's docstring. Single global slot: this is an offline,
        # single-session demo (one analyst, one capture at a time), not a
        # multi-tenant API, so there is no per-user/per-session keying here.
        _state["last_context"] = final_context
        filename = (csv_file.filename if csv_file else None) or (pcap_file.filename if pcap_file else None)
        return AnalyzeResponse(filename=filename, **result)

    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Pipeline failure during /analyze")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post(
    "/simulate",
    response_model=SimulateResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def simulate(request: SimulateRequest):
    """Phase 9, Feature 3 -- Counterfactual Intervention Simulator. Call
    /analyze first; this endpoint re-runs the rollout on that same context
    with a defender action applied, and returns the before/after comparison.
    No frontend UI for this yet (see CLAUDE.md's Phase 9 note) -- this is
    endpoint wiring only, exposed for direct API use / future frontend work.
    """
    bundle: ModelBundle = _state["bundle"]
    context = _state.get("last_context")
    if context is None:
        raise HTTPException(status_code=422, detail="No analyzed context available yet -- call /analyze first.")
    if request.action not in ("isolate_host", "block_flow"):
        raise HTTPException(status_code=422, detail=f"Unknown action {request.action!r}; expected 'isolate_host' or 'block_flow'.")

    try:
        result = run_simulation(
            bundle, context,
            action_kind=request.action, host_ip=request.host_ip,
            src_ip=request.src_ip, dst_ip=request.dst_ip,
            bidirectional=request.bidirectional, k=request.k,
        )
        return SimulateResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Pipeline failure during /simulate")
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")
