#!/usr/bin/env bash
# =============================================================================
# run_attendance_system.sh
# Launcher for the Face Recognition Attendance System
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- colours -----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Python ------------------------------------------------------------------
PYTHON=$(command -v python3 || command -v python || true)
[ -z "$PYTHON" ] && error "Python 3 not found. Install Python 3.10+."
info "Python: $($PYTHON --version)"

# --- PostgreSQL via Docker Compose -------------------------------------------
if command -v docker &>/dev/null; then
    info "Starting PostgreSQL container..."
    docker compose up -d 2>&1 | grep -v "^$" || true
    info "Waiting for PostgreSQL to be healthy..."
    for i in $(seq 1 20); do
        status=$(docker inspect attendance_postgres --format='{{.State.Health.Status}}' 2>/dev/null || echo "not_found")
        if [ "$status" = "healthy" ]; then
            info "PostgreSQL is healthy ✓"
            break
        fi
        [ "$i" -eq 20 ] && warn "Postgres may not be ready yet — continuing anyway"
        sleep 2
    done
else
    warn "Docker not found. Ensure PostgreSQL is running manually (see .env for connection details)."
fi

# --- Python dependencies -----------------------------------------------------
info "Installing/verifying Python dependencies..."
$PYTHON -m pip install -r requirements.txt --quiet

# --- Check dlib models -------------------------------------------------------
MODELS_DIR="data/data_dlib"
LANDMARKS="$MODELS_DIR/shape_predictor_68_face_landmarks.dat"
FACE_REC="$MODELS_DIR/dlib_face_recognition_resnet_model_v1.dat"
if [ ! -f "$LANDMARKS" ] || [ ! -f "$FACE_REC" ]; then
    warn "dlib model files not found in $MODELS_DIR/"
    warn "Download them from: http://dlib.net/files/"
    warn "  - shape_predictor_68_face_landmarks.dat.bz2"
    warn "  - dlib_face_recognition_resnet_model_v1.dat.bz2"
    warn "Decompress and place both .dat files in $MODELS_DIR/"
    read -rp "Press Enter to continue anyway, or Ctrl-C to abort..."
fi

# --- Launch FastAPI dashboard ------------------------------------------------
info "Launching FastAPI dashboard → http://localhost:8000"
info "API docs available at  → http://localhost:8000/docs"
exec $PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000
