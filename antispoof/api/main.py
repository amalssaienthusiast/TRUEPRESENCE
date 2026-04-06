"""
api/main.py — FastAPI Liveness Verification API

Endpoints:
  POST /api/v1/liveness/verify          — verify a frame for liveness
  GET  /api/v1/liveness/health          — model health check
  POST /api/v1/liveness/reset/{session_id} — reset blink state for a session

Session management: EAR gate state is maintained per session_id
in an in-memory dict with a TTL of 30 seconds.
"""

import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

# Add parent dir so imports work when running from antispoof/
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.middleware import setup_middleware
from api.schemas import HealthResponse, LivenessVerifyResponse, ResetResponse
from training.config import EARConfig, PipelineConfig
from pipeline.ear_gate import EyeBlinkLivenessGate
from pipeline.screen_gate import ScreenSpoofGate
from pipeline.spoof_gate import SpoofClassifierGate
from pipeline.antispoof_pipeline import AntiSpoofPipeline


# ---------------------------------------------------------------------------
# Session store  — {session_id: {"gate": EyeBlinkLivenessGate, "last_access": float}}
# ---------------------------------------------------------------------------

_SESSION_TTL: float = float(os.getenv("SESSION_TTL_SECONDS", "30"))
_sessions: Dict[str, dict] = {}


def _get_or_create_session(session_id: str, cfg: EARConfig) -> EyeBlinkLivenessGate:
    """Return the EAR gate for a session, creating it if new or expired."""
    now = time.time()
    # Expire old sessions
    expired = [sid for sid, s in _sessions.items() if now - s["last_access"] > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]

    if session_id not in _sessions:
        _sessions[session_id] = {
            "gate":        EyeBlinkLivenessGate(
                landmarks_path       = cfg.landmarks_path,
                ear_threshold        = cfg.ear_threshold,
                blink_consec_frames  = cfg.blink_consec_frames,
                required_blinks      = cfg.required_blinks,
                time_window_sec      = cfg.time_window_sec,
            ),
            "last_access": now,
        }
    else:
        _sessions[session_id]["last_access"] = now

    return _sessions[session_id]["gate"]


# ---------------------------------------------------------------------------
# Global pipeline (shared across requests — thread-safe for inference)
# ---------------------------------------------------------------------------

pipeline: AntiSpoofPipeline | None = None
pipeline_cfg: PipelineConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup."""
    global pipeline, pipeline_cfg
    pipeline_cfg = PipelineConfig(
        ear               = EARConfig(
            landmarks_path = os.getenv(
                "DLIB_LANDMARKS",
                "../data/data_dlib/shape_predictor_68_face_landmarks.dat",
            ),
        ),
        screen_weights    = os.getenv("SCREEN_CKPT", "checkpoints/best_screen_detector.pt"),
        antispoof_weights = os.getenv("ANTISPOOF_CKPT", "checkpoints/best_antispoof.onnx"),
        live_threshold    = 0.52,
        device            = "cpu",
    )
    pipeline = AntiSpoofPipeline(pipeline_cfg)
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "TruePresence Anti-Spoofing API",
    description = "Three-stage face liveness verification: EAR blink → screen detector → CNN classifier.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

setup_middleware(app)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to OpenAPI docs."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/liveness/verify",
    response_model = LivenessVerifyResponse,
    summary        = "Verify frame liveness through all 3 anti-spoofing stages",
)
async def verify_liveness(
    frame:      UploadFile = File(...,   description="JPEG or PNG webcam frame"),
    session_id: str        = Form(...,   description="Unique session identifier per user/attempt"),
):
    """Process one webcam frame through the 3-stage anti-spoofing pipeline.

    The EAR blink state is maintained per session_id.
    Multiple frames from the same session are required (PENDING → LIVE/SPOOF).
    """
    import cv2

    # Decode uploaded image
    contents = await frame.read()
    arr      = np.frombuffer(contents, dtype=np.uint8)
    img_bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return JSONResponse({"detail": "Invalid image — could not decode."}, status_code=400)

    # Use session-aware EAR gate instead of pipeline's shared gate
    if pipeline_cfg:
        session_gate = _get_or_create_session(session_id, pipeline_cfg.ear)
        # Temporarily swap the pipeline's Stage 1 gate
        original_gate = pipeline.stage1
        pipeline.stage1 = session_gate

    result = pipeline.run(img_bgr)

    if pipeline_cfg:
        pipeline.stage1 = original_gate   # restore

    response = LivenessVerifyResponse(
        verdict       = result.verdict,
        confidence    = result.confidence,
        blinks        = result.stage_results.get("ear", {}).get("blinks", 0),
        spoof_reason  = result.spoof_reason,
        stage_results = result.stage_results,
        latency_ms    = result.latency_ms,
        session_id    = session_id,
    )

    # Set custom headers for middleware logging
    headers = {
        "X-Verdict":      result.verdict,
        "X-Spoof-Reason": result.spoof_reason or "none",
        "X-Session-ID":   session_id,
    }
    return JSONResponse(content=response.model_dump(), headers=headers)


@app.get(
    "/api/v1/liveness/health",
    response_model = HealthResponse,
    summary        = "Check model load status and hardware info",
)
async def health():
    """Return the health status of each loaded model stage."""
    import torch

    stage2_ok = pipeline is not None and pipeline.stage2._model is not None
    stage3_ok = pipeline is not None and (
        pipeline.stage3._ort_session is not None or pipeline.stage3._torch_model is not None
    )
    stage1_ok = pipeline is not None and pipeline.stage1._predictor is not None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    status = "ok" if (stage1_ok and stage2_ok and stage3_ok) else "degraded"

    return HealthResponse(
        status            = status,
        device            = device,
        stage1_dlib       = stage1_ok,
        stage2_yolo       = stage2_ok,
        stage3_onnx       = stage3_ok,
        antispoof_weights = os.getenv("ANTISPOOF_CKPT", "checkpoints/best_antispoof.onnx"),
        screen_weights    = os.getenv("SCREEN_CKPT", "checkpoints/best_screen_detector.pt"),
    )


@app.post(
    "/api/v1/liveness/reset/{session_id}",
    response_model = ResetResponse,
    summary        = "Reset blink state for a session (call after attendance recorded)",
)
async def reset_session(session_id: str):
    """Clear the EAR blink counter for the given session.

    Call this after a successful attendance mark so the next person
    starts with a fresh blink window.
    """
    if session_id in _sessions:
        _sessions[session_id]["gate"].reset()
        _sessions[session_id]["last_access"] = time.time()
        msg = f"Session '{session_id}' reset successfully."
    else:
        msg = f"Session '{session_id}' not found — will be created on next verify call."

    return ResetResponse(session_id=session_id, message=msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
