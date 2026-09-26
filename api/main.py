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
from fastapi.staticfiles import StaticFiles

from api.pipeline import InsufficientDataError, ModelBundle, run_pipeline, run_simulation
from api.schemas import AnalyzeResponse, CalibrationResponse, ErrorResponse, ROCPRCurvesResponse, SimulateRequest, SimulateResponse
from models.train import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = "checkpoints/phase3"
DEFAULT_PIPELINE_CONFIG = "configs/pipeline.yaml"
ROC_PR_CURVES_PATH = Path("benchmark/roc_pr_curves.json")
CALIBRATION_PATH = Path("benchmark/calibration.json")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device("auto")
    logger.info(f"Loading world model from {DEFAULT_CHECKPOINT_DIR} onto {device} ...")
    _state["bundle"] = ModelBundle.load(DEFAULT_CHECKPOINT_DIR, DEFAULT_PIPELINE_CONFIG, device)
    logger.info("Model loaded.")

    # Fixed, precomputed MODEL-level evaluation data (see
    # benchmark/run_benchmark.py) -- loaded once at startup like the model
    # itself, never recomputed per request; independent of any /analyze call.
    if ROC_PR_CURVES_PATH.exists():
        _state["roc_pr_curves"] = json.loads(ROC_PR_CURVES_PATH.read_text())
        logger.info(f"Loaded ROC/PR curve data from {ROC_PR_CURVES_PATH}.")
    else:
        _state["roc_pr_curves"] = None
        logger.warning(f"{ROC_PR_CURVES_PATH} not found -- /model-performance/roc-pr will 404 until `python benchmark/run_benchmark.py` is run.")

    if CALIBRATION_PATH.exists():
        _state["calibration"] = json.loads(CALIBRATION_PATH.read_text())
        logger.info(f"Loaded calibration data from {CALIBRATION_PATH}.")
    else:
        _state["calibration"] = None
        logger.warning(f"{CALIBRATION_PATH} not found -- /model-performance/calibration will 404 until `python benchmark/run_benchmark.py` is run.")

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


class StripAPIPrefixMiddleware:
    """Deployment mirrors the Vite dev proxy's own rewrite
    (`path.replace(/^\\/api/, '')`) so the SAME frontend code -- which
    always calls `/api/analyze`, `/api/health`, etc. -- works unchanged in
    production, where there's no separate dev server to do that rewrite.
    In dev this middleware is simply never exercised, since Vite's proxy
    already strips the prefix before the request reaches this app; in
    prod (frontend served same-origin, see the StaticFiles mount at the
    bottom of this file) it does the identical job. Real API routes are
    untouched either way -- curl/`/docs` against the bare paths still work."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api/"):
            scope = dict(scope)
            scope["path"] = scope["path"][len("/api"):]
        await self.app(scope, receive, send)


app.add_middleware(StripAPIPrefixMiddleware)


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
    "/model-performance/roc-pr",
    response_model=ROCPRCurvesResponse,
    responses={404: {"model": ErrorResponse}},
)
def roc_pr_curves():
    """Full ROC/PR curve points for the held-out day (unseen attack type) --
    the real data behind benchmark/report.md's headline roc_auc figure, not
    a single operating point. Computed once by benchmark/run_benchmark.py,
    served here unchanged. Independent of any /analyze call: this describes
    the MODEL's fixed evaluation, not a particular upload."""
    curves = _state.get("roc_pr_curves")
    if curves is None:
        raise HTTPException(status_code=404, detail=f"No ROC/PR curve data found at {ROC_PR_CURVES_PATH} -- run `python benchmark/run_benchmark.py` first.")
    return ROCPRCurvesResponse(**curves)


@app.get(
    "/model-performance/calibration",
    response_model=CalibrationResponse,
    responses={404: {"model": ErrorResponse}},
)
def calibration():
    """Reliability diagram: does the model's predicted confidence match its
    real accuracy (e.g. are '70% confidence' predictions actually right
    ~70% of the time)? Held-out day, world model only. Computed once by
    benchmark/run_benchmark.py, served here unchanged."""
    data = _state.get("calibration")
    if data is None:
        raise HTTPException(status_code=404, detail=f"No calibration data found at {CALIBRATION_PATH} -- run `python benchmark/run_benchmark.py` first.")
    return CalibrationResponse(**data)


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


# Deployment (Docker/Hugging Face Spaces): serves the built frontend
# (frontend/dist, produced by `npm run build`) from this same process, so
# the console and the API share one origin -- no separate frontend host,
# no cross-origin request to configure. Registered LAST and deliberately
# guarded on the directory actually existing: local dev never builds
# frontend/dist (Vite's own dev server serves the frontend instead), so
# this mount is silently skipped there rather than erroring, and because
# it's added after every real @app.get/@app.post above, those routes are
# always matched first -- this only catches whatever's left over.
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
    logger.info(f"Serving built frontend from {_frontend_dist}.")
else:
    logger.info(f"{_frontend_dist} not found -- not serving a frontend (expected in local dev; run `npm run build` for a deployment build).")
