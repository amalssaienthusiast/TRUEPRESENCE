"""
api/middleware.py — CORS, rate limiting, and request logging middleware.

- CORS: allowed origins from .env ALLOWED_ORIGINS
- Rate limit: max 10 requests/second per IP (using slowapi)
- Logging: session_id, verdict, latency, spoof_reason → log file
"""

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

LOG_DIR = Path(os.getenv("LOG_DIR", "runs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

api_logger = logging.getLogger("antispoof.api")
api_logger.setLevel(logging.INFO)

# File handler for structured API logs
_fh = logging.FileHandler(LOG_DIR / "api_requests.log")
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
api_logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests: method, path, status, latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and log timing."""
        t0 = time.perf_counter()
        response = await call_next(request)
        latency  = (time.perf_counter() - t0) * 1000

        # Log liveness-specific extras from response headers (set by endpoint)
        session_id   = request.headers.get("X-Session-ID", "-")
        verdict      = response.headers.get("X-Verdict", "-")
        spoof_reason = response.headers.get("X-Spoof-Reason", "-")

        api_logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"latency={latency:.1f}ms "
            f"session={session_id} "
            f"verdict={verdict} "
            f"spoof_reason={spoof_reason}"
        )
        return response


# ---------------------------------------------------------------------------
# Rate limiting via slowapi
# ---------------------------------------------------------------------------

def get_limiter():
    """Return a configured slowapi Limiter instance."""
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        return Limiter(key_func=get_remote_address, default_limits=["10/second"])
    except ImportError:
        print("[Middleware] slowapi not installed — rate limiting disabled.")
        return None


# ---------------------------------------------------------------------------
# Setup function — called by api/main.py
# ---------------------------------------------------------------------------

def setup_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI app.

    Args:
        app: The FastAPI application instance.
    """
    # CORS
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = [o.strip() for o in allowed_origins],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Rate limiting
    limiter = get_limiter()
    if limiter:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        print("[Middleware] Rate limiting: 10 req/sec per IP")

    print(f"[Middleware] CORS allowed origins: {allowed_origins}")
