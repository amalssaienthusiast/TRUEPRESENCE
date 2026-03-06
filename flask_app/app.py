"""
flask_app/app.py — Web dashboard for the Face Recognition Attendance System

Uses PostgreSQL (via the shared database.py module) instead of SQLite.
The database is now the single source of truth — no more file copying.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import subprocess
import threading
import os
import sys
import signal
import time

# ---------------------------------------------------------------------------
# Database helpers (PostgreSQL)
# ---------------------------------------------------------------------------
# Resolve the parent project directory so we can import database.py
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from database import init_db, get_records_by_date, get_all_records  # noqa: E402

app = Flask(__name__)

# Absolute path to parent directory (for running Python scripts)
_PARENT_DIR = PARENT_DIR

# ---------------------------------------------------------------------------
# Global process state
# ---------------------------------------------------------------------------
current_process = None
process_output  = ""
process_name    = None


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

def run_script(script_name: str, auto_next: bool = False, next_script: str | None = None) -> bool:
    """Run a Python script in the parent directory and stream its output."""
    global current_process, process_output, process_name

    if current_process and current_process.poll() is None:
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        except Exception:
            pass

    process_output = ""
    process_name   = script_name
    script_path    = os.path.join(_PARENT_DIR, script_name)

    if not os.path.exists(script_path):
        process_output = f"Error: Script not found — {script_path}"
        return False

    try:
        # OPENCV_AVFOUNDATION_SKIP_AUTH=1 prevents OpenCV from trying to request
        # macOS camera permission itself (which fails from non-main threads).
        # The PyQt6 app handles permission via its own main-thread camera open.
        env = os.environ.copy()
        env["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

        current_process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=_PARENT_DIR,
            preexec_fn=os.setsid,
            env=env,
        )

        def monitor():
            global process_output
            for line in current_process.stdout:
                process_output += line
                print(line, end="", flush=True)
            if auto_next and next_script and current_process.poll() is not None:
                time.sleep(1)
                run_script(next_script)

        t = threading.Thread(target=monitor, daemon=True)
        t.start()
        return True

    except Exception as e:
        process_output = f"Error starting {script_name}: {e}"
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/favicon.ico")
def favicon():
    return "", 204  # no favicon — silence browser 404 noise


@app.route("/static/css/custom.css")
def custom_css():
    return "", 204  # placeholder — silence browser/extension 404 noise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_script", methods=["POST"])
def execute_script():
    script = request.form.get("script")

    if script == "get_faces":
        ok = run_script("get_faces_from_camera.py", auto_next=True, next_script="features_extraction_to_csv.py")
    elif script == "extract_features":
        ok = run_script("features_extraction_to_csv.py")
    elif script == "attendance":
        ok = run_script("attendance_taker.py")
    else:
        return jsonify({"status": "error", "message": "Invalid script specified"})

    if ok:
        return jsonify({"status": "success", "message": f"Started {script}"})
    return jsonify({"status": "error", "message": process_output})


@app.route("/status")
def get_status():
    return jsonify({
        "running": current_process is not None and current_process.poll() is None,
        "output":  process_output,
        "script":  process_name,
    })


@app.route("/stop_script", methods=["POST"])
def stop_script():
    global current_process
    if current_process and current_process.poll() is None:
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            return jsonify({"status": "success", "message": "Process terminated"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "No process running"})


@app.route("/attendance")
def view_attendance():
    try:
        records = get_records_by_date()
    except Exception as e:
        records = {"error": str(e)}
    return render_template("attendance.html", records=records)


@app.route("/api/attendance")
def attendance_api():
    try:
        records = get_records_by_date()
    except Exception as e:
        records = {"error": str(e)}
    return jsonify(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        init_db()
        print("✓ Database schema ready.")
    except Exception as e:
        print(f"⚠  Database connection failed: {e}")
        print("   Ensure PostgreSQL is running:  docker compose up -d")

    app.run(debug=True, threaded=True)
