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

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.pipeline import InsufficientDataError, ModelBundle, run_pipeline
from api.schemas import AnalyzeResponse, ErrorResponse
from models.train import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = "checkpoints/phase3"
DEFAULT_PIPELINE_CONFIG = "configs/pipeline.yaml"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device("auto")
    logger.info(f"Loading world model from {DEFAULT_CHECKPOINT_DIR} onto {device} ...")
    _state["bundle"] = ModelBundle.load(DEFAULT_CHECKPOINT_DIR, DEFAULT_PIPELINE_CONFIG, device)
    logger.info("Model loaded.")
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

        result = run_pipeline(bundle, csv_path, pcap_path, k=k)
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
