"""
api/schemas.py — Pydantic request/response models for the liveness API.

All verdict fields use Literal types for strict validation.
"""

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Verdict type alias
# ---------------------------------------------------------------------------

VerdictType = Literal["LIVE", "SPOOF", "PENDING", "NO_FACE"]


# ---------------------------------------------------------------------------
# Stage result sub-models
# ---------------------------------------------------------------------------

class EARStageResult(BaseModel):
    """Result from Stage 1 — Eye Aspect Ratio blink gate."""
    live:      Optional[bool]  = None
    blinks:    int             = 0
    ear:       float           = 0.0
    time_left: float           = 0.0


class ScreenStageResult(BaseModel):
    """Result from Stage 2 — Screen/phone object detector."""
    spoof_detected: bool          = False
    detections:     list[dict]    = Field(default_factory=list)
    verdict:        str           = "PASS"
    reason:         Optional[str] = None


class ClassifierStageResult(BaseModel):
    """Result from Stage 3 — CNN real vs fake classifier."""
    live_prob:  float = 0.5
    spoof_prob: float = 0.5
    verdict:    str   = "SPOOF"
    passed:     bool  = False


class StageResults(BaseModel):
    """Combined stage results container."""
    ear:        EARStageResult        = Field(default_factory=EARStageResult)
    screen:     ScreenStageResult     = Field(default_factory=ScreenStageResult)
    classifier: ClassifierStageResult = Field(default_factory=ClassifierStageResult)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class LivenessVerifyResponse(BaseModel):
    """Response model for POST /api/v1/liveness/verify."""
    verdict:       VerdictType
    confidence:    float               = Field(ge=0.0, le=1.0)
    blinks:        int                 = 0
    spoof_reason:  Optional[str]       = None
    stage_results: Dict[str, Any]      = Field(default_factory=dict)
    latency_ms:    float               = 0.0
    session_id:    str


class HealthResponse(BaseModel):
    """Response model for GET /api/v1/liveness/health."""
    status:           Literal["ok", "degraded"]
    device:           str
    stage1_dlib:      bool
    stage2_yolo:      bool
    stage3_onnx:      bool
    antispoof_weights: str
    screen_weights:   str


class ResetResponse(BaseModel):
    """Response model for POST /api/v1/liveness/reset/{session_id}."""
    session_id: str
    message:    str
