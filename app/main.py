"""
app/main.py — FastAPI web dashboard for TruePresence (SQLite edition)

Routes:
  GET  /                          Dashboard
  POST /run_script                Launch a subprocess
  GET  /status                    Poll running process
  POST /stop_script               Kill running process
  GET  /attendance                Attendance records page
  GET  /api/attendance            Records as JSON
  GET  /api/stats                 Dashboard stats (total/today/valid/invalid/db_ok)
  GET  /api/attendance/export     CSV download
  GET  /health                    System health check (always returns 200)

Resilience principles:
  - DB errors never crash the server — all DB calls are try/except in database.py
  - Subprocess errors return JSON {status: "error", message: "..."}, never 500
  - If DB is unavailable on startup, app continues with degraded DB features
  - Output buffer capped at 300 lines to prevent memory growth
"""

import csv
import io
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("truepresence.dashboard")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR    = Path(__file__).parent.resolve()
ROOT_DIR   = APP_DIR.parent                  # project root

# Import DB layer from project root
sys.path.insert(0, str(ROOT_DIR))
try:
    from database import init_db, get_records_by_date, get_stats, export_csv_rows, get_all_records
    _db_imported = True
except ImportError as _exc:
    logger.warning("database.py import failed (%s) — DB features disabled.", _exc)
    _db_imported = False
    # Provide no-op stubs so the rest of the file compiles cleanly
    def init_db():         return False          # noqa: E306
    def get_records_by_date(): return {}         # noqa: E306
    def get_stats():       return {"total":0,"today":0,"valid":0,"invalid":0,"db_ok":False}  # noqa: E306
    def export_csv_rows(): return [["Name","Time","Date","Status"]]  # noqa: E306
    def get_all_records(): return []             # noqa: E306

# ---------------------------------------------------------------------------
# Templates + static files
# ---------------------------------------------------------------------------
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# ---------------------------------------------------------------------------
# Subprocess state (one process at a time)
# ---------------------------------------------------------------------------
_proc_lock: threading.Lock = threading.Lock()
current_process: subprocess.Popen | None = None
_output_lines:   list[str]               = []    # capped at 300 lines
process_name:    str | None              = None
_MAX_OUTPUT_LINES: int                   = 300


def _get_output() -> str:
    return "".join(_output_lines)


def _append_output(line: str) -> None:
    """Append a line to the output buffer, evicting old lines if over limit."""
    _output_lines.append(line)
    while len(_output_lines) > _MAX_OUTPUT_LINES:
        _output_lines.pop(0)


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

def run_script(
    script_name: str,
    auto_next: bool = False,
    next_script: str | None = None,
) -> bool:
    """Launch *script_name* from the project root as a subprocess.

    Returns True if the process started successfully.
    """
    global current_process, _output_lines, process_name

    # Kill any already-running process without raising
    with _proc_lock:
        if current_process and current_process.poll() is None:
            try:
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
                current_process.wait(timeout=3)
            except Exception as exc:
                logger.warning("Could not kill previous process: %s", exc)

        _output_lines = []
        process_name  = script_name
        script_path   = ROOT_DIR / script_name

    if not script_path.exists():
        _append_output(f"[ERROR] Script not found: {script_path}\n")
        logger.error("Script not found: %s", script_path)
        return False

    try:
        env = os.environ.copy()
        env["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"   # macOS camera safety

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout     = subprocess.PIPE,
            stderr     = subprocess.STDOUT,
            text       = True,
            cwd        = str(ROOT_DIR),
            preexec_fn = os.setsid,
            env        = env,
        )

        with _proc_lock:
            current_process = proc

        def _monitor() -> None:
            try:
                for line in proc.stdout:
                    _append_output(line)
                    print(line, end="", flush=True)
            except Exception as exc:
                logger.warning("Output monitor error: %s", exc)
            finally:
                if auto_next and next_script and proc.poll() is not None:
                    time.sleep(1)
                    run_script(next_script)

        threading.Thread(target=_monitor, daemon=True, name=f"monitor-{script_name}").start()
        logger.info("Started subprocess: %s (pid=%d)", script_name, proc.pid)
        return True

    except Exception as exc:
        _append_output(f"[ERROR] Failed to start {script_name}: {exc}\n")
        logger.error("Failed to start %s: %s", script_name, exc)
        return False


# ---------------------------------------------------------------------------
# Lifespan — initialise DB once on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if _db_imported:
        success = init_db()
        if success:
            logger.info("✓ SQLite database ready.")
        else:
            logger.warning(
                "⚠  SQLite init failed. "
                "Check DB_PATH in .env (default: data/attendance.db). "
                "Attendance recording will not persist until resolved."
            )
    else:
        logger.warning("⚠  database.py not imported — DB features are disabled.")
    yield
    # Cleanup: terminate any running subprocess on shutdown
    if current_process and current_process.poll() is None:
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "TruePresence — Face Recognition Attendance System",
    description = "Control panel and attendance viewer. SQLite backend, zero external dependencies.",
    version     = "2.1.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Routes — dashboard pages
# ---------------------------------------------------------------------------

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={}, status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main control-panel dashboard."""
    try:
        stats = get_stats()
    except Exception:
        stats = {"total": 0, "today": 0, "valid": 0, "invalid": 0, "db_ok": False}
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})


@app.get("/attendance", response_class=HTMLResponse)
async def view_attendance(request: Request):
    """Render the attendance records page."""
    try:
        records = get_records_by_date()
    except Exception as exc:
        logger.warning("view_attendance DB error: %s", exc)
        records = {}
        return templates.TemplateResponse(
            "attendance.html",
            {"request": request, "records": {}, "error": str(exc)},
        )
    return templates.TemplateResponse(
        "attendance.html", {"request": request, "records": records, "error": None}
    )


# ---------------------------------------------------------------------------
# Routes — process control
# ---------------------------------------------------------------------------

@app.post("/run_script")
async def execute_script(script: str = Form(...)):
    """Launch a Python script by logical name.

    Accepted values: get_faces | extract_features | attendance
    """
    SCRIPT_MAP = {
        "get_faces":        ("get_faces_from_camera.py",     True,  "features_extraction_to_csv.py"),
        "extract_features": ("features_extraction_to_csv.py", False, None),
        "attendance":       ("attendance_taker.py",           False, None),
    }

    if script not in SCRIPT_MAP:
        return JSONResponse(
            {"status": "error", "message": f"Unknown script '{script}'. "
             f"Valid options: {', '.join(SCRIPT_MAP)}"}, status_code=400
        )

    target, auto_next, next_sc = SCRIPT_MAP[script]
    ok = run_script(target, auto_next=auto_next, next_script=next_sc)
    if ok:
        return JSONResponse({"status": "success", "message": f"Started '{target}'"})
    return JSONResponse(
        {"status": "error", "message": _get_output() or f"Failed to start {target}"},
        status_code=500,
    )


@app.get("/status")
async def get_process_status():
    """Return current subprocess status and buffered stdout."""
    running = current_process is not None and current_process.poll() is None
    return {
        "running":    running,
        "output":     _get_output(),
        "script":     process_name,
        "exit_code":  current_process.poll() if current_process else None,
    }


@app.post("/stop_script")
async def stop_script():
    """Send SIGTERM to the running process group."""
    global current_process
    if current_process and current_process.poll() is None:
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            logger.info("Sent SIGTERM to process '%s'", process_name)
            return {"status": "success", "message": "Process terminated"}
        except Exception as exc:
            logger.warning("stop_script error: %s", exc)
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)
    return JSONResponse({"status": "error", "message": "No process is currently running"}, status_code=404)


# ---------------------------------------------------------------------------
# Routes — data API
# ---------------------------------------------------------------------------

@app.get("/api/attendance")
async def attendance_api():
    """Return all attendance records grouped by date as JSON."""
    try:
        return get_records_by_date()
    except Exception as exc:
        logger.warning("attendance_api error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.get("/api/stats")
async def stats_api():
    """Return dashboard summary stats: total, today, valid, invalid, db_ok."""
    try:
        return get_stats()
    except Exception as exc:
        logger.warning("stats_api error: %s", exc)
        return {"total": 0, "today": 0, "valid": 0, "invalid": 0, "db_ok": False}


@app.get("/api/attendance/export")
async def export_csv():
    """Stream all attendance records as a CSV file download."""
    try:
        rows = export_csv_rows()
    except Exception as exc:
        logger.warning("export_csv error: %s", exc)
        rows = [["Name", "Time", "Date", "Status"]]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    buf.seek(0)

    today = __import__("datetime").date.today().isoformat()
    filename = f"attendance_{today}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type = "text/csv",
        headers    = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
async def health():
    """Always returns 200. Lets load-balancers / monitors check the app is alive."""
    try:
        stats = get_stats()
        db_ok = stats.get("db_ok", False)
    except Exception:
        db_ok = False
    return {
        "status":  "ok",
        "db_ok":   db_ok,
        "db_path": DB_PATH if _db_imported else "N/A",
        "version": "2.1.0",
    }


# Re-export DB_PATH for /health endpoint
try:
    from database import DB_PATH
except ImportError:
    DB_PATH = "N/A"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
