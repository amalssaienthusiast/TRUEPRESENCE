# TruePresence — Face Recognition Attendance System

> **Spoof-proof, blink-aware, liveness-verified attendance — powered by dlib, PyQt6 and PostgreSQL.**

TruePresence replaces paper registers and PIN-based check-ins with a multi-layer biometric attendance system that actively rejects printed photos, screen replays, and proxy attempts.  Every attendance record is stored in a PostgreSQL database running in Docker, and a Flask web dashboard gives you a real-time view of records from any browser on the same machine.

---

## Table of Contents

1. [Feature Overview](#1-feature-overview)
2. [Architecture](#2-architecture)
3. [Anti-Spoofing Pipeline](#3-anti-spoofing-pipeline)
4. [Project Structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Quick Start](#6-quick-start)
7. [Step-by-Step Setup](#7-step-by-step-setup)
   - 7.1 [Clone & Install](#71-clone--install-dependencies)
   - 7.2 [Download dlib Models](#72-download-dlib-models)
   - 7.3 [Start PostgreSQL](#73-start-postgresql-docker)
   - 7.4 [Register Faces](#74-register-faces)
   - 7.5 [Extract Features](#75-extract-features)
   - 7.6 [Take Attendance](#76-take-attendance)
   - 7.7 [Web Dashboard](#77-web-dashboard)
8. [GUI Reference](#8-gui-reference)
   - 8.1 [Face Registration Window](#81-face-registration-window)
   - 8.2 [Attendance Window](#82-attendance-window)
   - 8.3 [Flask Dashboard](#83-flask-dashboard)
9. [Database Schema](#9-database-schema)
10. [Configuration (.env)](#10-configuration-env)
11. [Flask API Reference](#11-flask-api-reference)
12. [How Attendance Status Is Decided](#12-how-attendance-status-is-decided)
13. [Troubleshooting](#13-troubleshooting)
14. [Security Notes](#14-security-notes)
15. [Contributing](#15-contributing)

---

## 1. Feature Overview

| Capability | Detail |
|---|---|
| **Face Registration** | PyQt6 desktop GUI — capture N face photos per person via live camera |
| **128-D Face Encoding** | dlib ResNet-50 produces a 128-dimensional descriptor per face |
| **Real-time Recognition** | Euclidean-distance matching at 30 FPS; threshold 0.4 |
| **Eye Blink Detection** | Eye Aspect Ratio (EAR) < 0.30 triggers a confirmed blink |
| **Head Nod Detection** | Nose-tip vertical displacement detects genuine nodding |
| **Motion Liveness** | Landmark-variance over 10 frames; video-loop fingerprint rejected |
| **Texture Liveness** | HOG descriptor variance — printed photos and screens fail this test |
| **Challenge-Response** | Random BLINK or NOD challenge must be completed within 50 frames |
| **Composite Score** | Exponentially-smoothed score ≥ 50 → `VALID`; below → `INVALID` |
| **PostgreSQL Storage** | All attendance records in a containerised Postgres 15 database |
| **Flask Dashboard** | Browser-based view with filtering, date grouping and CSV export |
| **Docker Compose** | One-command database startup; named volume persists data |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User's Machine                        │
│                                                              │
│  ┌─────────────────────┐     ┌────────────────────────────┐  │
│  │  get_faces_from_    │     │    attendance_taker.py     │  │
│  │  camera.py          │     │                            │  │
│  │  (PyQt6 GUI)        │     │  Face_Recognizer           │  │
│  │                     │     │  ├─ dlib detector          │  │
│  │  ┌───────────────┐  │     │  ├─ ResNet encoder         │  │
│  │  │  OpenCV cam   │  │     │  ├─ Anti-spoof engine      │  │
│  │  │  dlib detect  │  │     │  │   ├─ EAR blink          │  │
│  │  │  Save ROI JPG │  │     │  │   ├─ Motion analysis    │  │
│  │  └───────────────┘  │     │  │   ├─ HOG texture        │  │
│  │         │           │     │  │   └─ Challenge-response │  │
│  └─────────┼───────────┘     │  └─ Liveness score (0-100)│  │
│            │                 │                            │  │
│            ▼                 │  VideoThread (QThread)     │  │
│  data/data_faces_from_       │  └─ pyqtSignal → PyQt6 UI │  │
│  camera/person_N_Name/       └─────────────┬──────────────┘  │
│            │                               │                  │
│            ▼                               ▼                  │
│  features_extraction_         ┌─────────────────────┐        │
│  to_csv.py                    │    database.py       │        │
│  └─ 128-D mean per person     │  psycopg2 pool       │        │
│     → data/features_all.csv   │  record_attendance() │        │
│                               └─────────┬───────────┘        │
│                                         │                     │
│  ┌──────────────────────┐               ▼                     │
│  │   flask_app/app.py   │    ┌───────────────────────┐       │
│  │   Flask dashboard    │───▶│  PostgreSQL 15         │       │
│  │   http://localhost:  │    │  (Docker container)    │       │
│  │   5000               │    │  attendance_db         │       │
│  └──────────────────────┘    └───────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### Data flow

```
Camera frames
    │
    ▼
dlib frontal face detector (HOG)
    │  bounding boxes
    ▼
dlib 68-point landmark predictor
    │  (x,y) of 68 facial points
    ├──► Eye Aspect Ratio  ──► blink detection
    ├──► Nose-tip positions ──► nod detection
    ├──► Landmark variance  ──► motion liveness
    └──► Face ROI patch     ──► HOG texture liveness
    │
    ▼
dlib ResNet-50 face descriptor (128-D vector)
    │
    ▼
Euclidean distance vs. known faces in features_all.csv
    │  best match < 0.4  → identified name
    │  no match          → "unknown"
    ▼
Composite liveness score (0–100, EMA-smoothed)
    │  ≥ 50 → VALID
    │  < 50 → INVALID + reason
    ▼
database.py → PostgreSQL attendance table
```

---

## 3. Anti-Spoofing Pipeline

TruePresence uses **four independent layers** whose scores are combined into a single smoothed value.

### Layer 1 — Eye Blink (EAR)

The Eye Aspect Ratio measures how open the eye is:

```
        |p2-p6| + |p3-p5|
EAR  =  ─────────────────
             2 · |p1-p4|
```

Where `p1–p6` are the six landmarks around one eye (dlib indices 36–41 for right, 42–47 for left).

- EAR < **0.30** for ≥ 1 consecutive frame → blink confirmed
- A live person blinks ~15–20 times/minute; a static photo never blinks

### Layer 2 — Motion Analysis (Landmark Variance)

Every frame, 13 key landmarks (eyes + nose tip) are compared to the previous frame.  A deque of 10 motion magnitudes is maintained per face.

| Condition | Decision |
|---|---|
| `avg_motion > 0.05` AND `variance > 0.01` | Real motion → score |
| `variance < 0.001` across full window | Video loop fingerprint → SPOOF |
| Motion too low | Static image → SPOOF |

### Layer 3 — Texture Analysis (HOG Descriptor)

The face ROI is resized to 64×64, converted to grayscale, and a HOG descriptor is computed (cell 8×8, block 16×16, 9 bins).  The variance of the descriptor values distinguishes:

- **Real skin** — complex, high-variance texture
- **Printed photo / screen** — smoother, low-variance texture (threshold: `variance * 1000 > 0.5`)

This check runs every **5 frames** to spare CPU.

### Layer 4 — Challenge-Response

On first detection, a random challenge is assigned:

| Challenge | Pass Condition |
|---|---|
| `BLINK` | EAR drops below threshold (confirmed blink) |
| `NOD` | Nose-tip vertical displacement ≥ 5 px with direction reversal |

The challenge must be completed within **50 frames** (~1.7 s at 30 FPS).  Completing it awards 20 liveness points.

### Composite Score

```
raw_score  =  (challenge ? 20 : 0)
           +  min(40, motion_value × 200)
           +  min(40, HOG_texture_score)       ← every 5th frame

smoothed   =  0.30 × raw_score
           +  0.70 × previous_smoothed_score   ← exponential moving average
```

- `smoothed ≥ 50` → **VALID** — status `"VALID"` written to DB
- `smoothed < 50` after 30 frames → **INVALID** — status includes the failure reason:
  - `INVALID - SPOOFING DETECTED`
  - `INVALID - Failed Liveness Check`
  - `INVALID - Challenge Not Completed`
  - `INVALID - No Blink Detected` *(only when `BLINK_REQUIRED = True`)*

---

## 4. Project Structure

```
.
├── attendance_taker.py          # PyQt6 attendance app (recognition + anti-spoof)
├── get_faces_from_camera.py     # PyQt6 face registration app
├── features_extraction_to_csv.py# Batch feature extraction → data/features_all.csv
├── database.py                  # PostgreSQL abstraction (psycopg2 pool)
├── docker-compose.yml           # PostgreSQL 15-alpine service definition
├── requirements.txt             # All Python dependencies
├── run_attendance_system.sh     # One-click launcher (Docker + Flask)
├── .env                         # Database connection config (git-ignored)
├── .env.example                 # Template — copy to .env
│
├── data/
│   ├── data_dlib/               # dlib model files (download separately)
│   │   ├── shape_predictor_68_face_landmarks.dat
│   │   └── dlib_face_recognition_resnet_model_v1.dat
│   ├── data_faces_from_camera/  # Captured face images, one folder per person
│   │   └── person_1_Alice/
│   │       ├── face_20260305_090000_1.jpg
│   │       └── …
│   └── features_all.csv         # 129-column CSV (name + 128 features)
│
├── flask_app/
│   ├── app.py                   # Flask web server
│   ├── templates/
│   │   ├── index.html           # Dashboard
│   │   └── attendance.html      # Records viewer
│   └── static/
│       ├── css/styles.css
│       └── js/main.js
│
├── CONTRIBUTING.md
├── SECURITY.md
└── .github/
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

---

## 5. Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 / 3.12 recommended |
| Docker Desktop | 24 | For PostgreSQL container |
| cmake | any | Required to compile dlib |
| C++ compiler | any | Xcode CLT on macOS; `build-essential` on Linux |
| Camera | any | Built-in webcam or USB camera |
| RAM | 4 GB | 8 GB recommended for dlib compilation |

### macOS
```bash
brew install cmake
xcode-select --install
```

### Ubuntu / Debian
```bash
sudo apt install cmake build-essential libopenblas-dev liblapack-dev libx11-dev
```

---

## 6. Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/Face-Recognition-Based-Attendance-System.git
cd Face-Recognition-Based-Attendance-System

# 2. Copy env config
cp .env.example .env          # edit if your Postgres runs elsewhere

# 3. Install Python deps
pip install -r requirements.txt

# 4. Place dlib model files in data/data_dlib/  (see §7.2)

# 5. Start everything with one command
./run_attendance_system.sh    # starts Docker Postgres + Flask dashboard
```

Then open **http://localhost:5000** and follow the three-step workflow in the dashboard.

---

## 7. Step-by-Step Setup

### 7.1 Clone & Install Dependencies

```bash
git clone https://github.com/your-org/Face-Recognition-Based-Attendance-System.git
cd Face-Recognition-Based-Attendance-System

# (optional) create a virtual environment
python3 -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

pip install -r requirements.txt
```

**What gets installed:**

| Package | Purpose |
|---|---|
| `dlib` | Face detection, landmarks, ResNet encoding |
| `opencv-python` | Camera capture, image processing |
| `numpy` | Numerical arrays, distance math |
| `scipy` | Euclidean distance helpers |
| `pandas` | Reading/writing features_all.csv |
| `scikit-image` | Image utilities |
| `scikit-learn` | HOG descriptor helpers |
| `imutils` | FPS counter, convenience wrappers |
| `PyQt6` | Desktop GUI (replaces tkinter) |
| `psycopg2-binary` | PostgreSQL driver |
| `python-dotenv` | Loads `.env` at runtime |
| `flask` | Web dashboard |

> **Note on dlib compilation:** dlib compiles a C++ shared library on first install.  This takes 5–15 minutes.  Ensure `cmake` and a C++ compiler are installed first.

---

### 7.2 Download dlib Models

Two pre-trained model files are required. They are **not** included in the repository due to size.

```bash
mkdir -p data/data_dlib
cd data/data_dlib

# 68-point landmark predictor (~100 MB)
curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# ResNet-50 face recognition model (~22 MB)
curl -LO http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2

cd ../..
```

Verify:
```
data/data_dlib/
├── shape_predictor_68_face_landmarks.dat       (99 MB)
└── dlib_face_recognition_resnet_model_v1.dat   (21 MB)
```

---

### 7.3 Start PostgreSQL (Docker)

```bash
docker compose up -d
```

This pulls **postgres:15-alpine**, creates:
- Database: `attendance_db`
- User: `attendance_user`  /  Password: `attendance_pass`
- Port: `5432` on localhost
- Named volume: `postgres_data` (data persists across container restarts)

Check health:
```bash
docker compose ps
# NAME                  STATUS
# attendance_postgres   Up X seconds (healthy)
```

The schema (single `attendance` table) is created automatically on first run of `attendance_taker.py` or `flask_app/app.py` via `database.init_db()`.

To stop (data preserved):
```bash
docker compose down
```

To stop and wipe all data:
```bash
docker compose down -v
```

---

### 7.4 Register Faces

```bash
python3 get_faces_from_camera.py
```

The **Face Registration** window opens:

1. **Step 1 — Create Folder**
   - Enter a unique numeric **ID** (e.g. `1`)
   - Enter the person's **Full Name** (e.g. `Alice Smith`)
   - Click **Create Person Folder**
   - A folder `data/data_faces_from_camera/person_1_Alice_Smith/` is created

2. **Step 2 — Capture Images**
   - Position face inside the green guide box
   - Press **Space** (or the Capture button) to save the current face crop
   - Capture **at least 10 images** per person; more = better accuracy
   - Vary lighting, slight head angles, expression
   - Green box = face in good position; Red box = face too close to edge

3. Repeat for each person to register.

> The window shows live FPS, face count, and an activity log. All images are saved as JPEG with timestamps.

---

### 7.5 Extract Features

```bash
python3 features_extraction_to_csv.py
```

This iterates every `person_*` folder in `data/data_faces_from_camera/`, computes a 128-D dlib descriptor for each image, averages them per person, and writes a single row to `data/features_all.csv`.

```
# features_all.csv format:
Alice_Smith, 0.0234, -0.1123, ..., 0.0871   (129 columns)
Bob_Jones,   0.1543,  0.0091, ..., -0.0432
```

**Re-run this whenever you register new people.**

---

### 7.6 Take Attendance

```bash
python3 attendance_taker.py
```

The **Attendance Window** opens (requires `data/features_all.csv` and a running PostgreSQL):

- Live camera feed with coloured bounding boxes:
  - 🟩 **Green** = live, recognised, VALID liveness score
  - 🟥 **Red** = spoofing detected or liveness failed
  - ⬜ **White** = face detected but not yet identified

- On-screen metrics per detected face:
  - Face ID, Liveness score, Blink status
  - EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), Motion, Nose Length
  - Active challenge and progress (`BLINK 3/10`, `NOD 7/10` …)
  - Spoof status message

- Attendance is written to PostgreSQL when a face is first recognised each day
- Status upgrades from `INVALID → VALID` if liveness is later confirmed

---

### 7.7 Web Dashboard

```bash
cd flask_app
python3 app.py
# → http://localhost:5000
```

Or use the launcher (starts Docker + Flask automatically):
```bash
./run_attendance_system.sh
```

The dashboard has four action cards:
| Card | Action |
|---|---|
| **Register Faces** | Launches `get_faces_from_camera.py` as a subprocess |
| **Extract Features** | Runs `features_extraction_to_csv.py` |
| **Take Attendance** | Launches `attendance_taker.py` |
| **View Records** | Opens the attendance records page |

Real-time console output is streamed to the browser while a subprocess is running.

---

## 8. GUI Reference

### 8.1 Face Registration Window

```
┌──────────────────────────────────────────────────────────┐
│  Camera Feed                    │ System Information      │
│  ┌──────────────────────────┐   │  FPS:              30.0 │
│  │                          │   │  Registered Persons: 3  │
│  │   ┌──────────────┐       │   │  Faces Detected:    1   │
│  │   │  (face box)  │       │   ├─────────────────────────┤
│  │   │   +          │       │   │ Registration Steps      │
│  │   └──────────────┘       │   │  ID:  [1          ]     │
│  │                          │   │  Name:[Alice Smith]     │
│  └──────────────────────────┘   │  [Create Person Folder] │
│                                 │  [Capture Face  Space ] │
│                                 │  Captured: 8 images     │
│                                 ├─────────────────────────┤
│                                 │ Activity Log            │
│                                 │  [09:12:34] Folder ready│
│                                 │  [09:12:41] Saved face_…│
└──────────────────────────────────────────────────────────┘
```

**Keyboard shortcut:** `Space` captures the current face.

### 8.2 Attendance Window

```
┌──────────────────────────────────────────────────────────────┐
│  Face Recognition  ·  Anti-Spoofing Attendance               │
├──────────────────────────────────┬───────────────────────────┤
│  Live Camera                     │ System Info               │
│  ┌──────────────────────────┐    │  FPS:      29.8           │
│  │ ┌────────────────────┐   │    │  Frame:    1204           │
│  │ │  [GREEN BOX] Alice │   │    │  Faces:    1              │
│  │ │  ID: Alice_Smith   │   │    │  Blinks:   3              │
│  │ │  Live: 72.4%       │   │    ├───────────────────────────┤
│  │ │  Blink: True  ✓    │   │    │ Face Metrics              │
│  │ │  EAR: 0.312        │   │    │  Face ID:     Alice_Smith │
│  │ │  MAR: 0.041        │   │    │  Liveness:    72.4        │
│  │ │  Motion: 0.08      │   │    │  Blinked:     True        │
│  │ │  Challenge:BLINK ✓ │   │    │  EAR:         0.312       │
│  │ └────────────────────┘   │    │  Challenge:   BLINK (10/10│
│  └──────────────────────────┘    │  Spoof Status: Live       │
│                                  ├───────────────────────────┤
│                                  │ Anti-Spoofing Active      │
│                                  │  ✅ Eye Blink (EAR)       │
│                                  │  ✅ Motion Analysis       │
│                                  │  ✅ HOG Texture           │
│                                  │  ✅ Challenge-Response    │
│                                  │  ✅ Composite Score ≥50   │
├──────────────────────────────────┴───────────────────────────┤
│  [  ✖  Quit  ]                                               │
└──────────────────────────────────────────────────────────────┘
```

### 8.3 Flask Dashboard

Navigate to **http://localhost:5000**:

- **Dashboard** — four action cards, real-time subprocess console
- **Attendance Records** (`/attendance`) — filterable table by name, status, date; CSV export button
- **API** (`/api/attendance`) — JSON endpoint for programmatic access

---

## 9. Database Schema

Single table in the `attendance_db` PostgreSQL database:

```sql
CREATE TABLE attendance (
    id      SERIAL  PRIMARY KEY,
    name    TEXT    NOT NULL,          -- matched person name from features_all.csv
    time    TEXT    NOT NULL,          -- HH:MM:SS of attendance event
    date    DATE    NOT NULL,          -- YYYY-MM-DD
    status  TEXT    NOT NULL           -- see §12 for possible values
             DEFAULT 'UNKNOWN',
    UNIQUE(name, date)                 -- one record per person per day
);
```

**Upgrade logic:** If a person is first marked `INVALID` (spoof detected) but then passes liveness later the same day, their record is **upgraded** to `VALID`.  It never downgrades (`VALID → INVALID` is blocked).

### Useful queries

```sql
-- Today's attendance
SELECT name, time, status
FROM   attendance
WHERE  date = CURRENT_DATE
ORDER  BY time;

-- All spoofing attempts
SELECT * FROM attendance
WHERE  status LIKE 'INVALID%'
ORDER  BY date DESC;

-- Attendance rate per person (last 30 days)
SELECT name,
       COUNT(*) FILTER (WHERE status = 'VALID') AS valid_days,
       COUNT(*) AS total_days
FROM   attendance
WHERE  date >= CURRENT_DATE - 30
GROUP  BY name;
```

---

## 10. Configuration (.env)

Copy `.env.example` to `.env` before first run.

```ini
DB_HOST=localhost       # Postgres hostname (use 'postgres' inside Docker network)
DB_PORT=5432            # Postgres port
DB_NAME=attendance_db   # Database name
DB_USER=attendance_user # Database user
DB_PASSWORD=attendance_pass  # Database password
```

These values match the `docker-compose.yml` defaults exactly — no editing required for local development.

To use an external PostgreSQL server, update all five values and ensure `psycopg2` can reach the host.

---

## 11. Flask API Reference

### `GET /`
Returns the main dashboard HTML page.

---

### `POST /run_script`
Launches one of the three Python scripts as a background subprocess.

**Form parameter:** `script` — one of `get_faces`, `extract_features`, `attendance`

**Response:**
```json
{ "status": "success", "message": "Started attendance" }
```

---

### `GET /status`
Polls the state of the currently running subprocess.

**Response:**
```json
{
  "running": true,
  "script":  "attendance_taker.py",
  "output":  "Frame 120 starts\nAlice_Smith: VALID\n..."
}
```

---

### `POST /stop_script`
Sends `SIGTERM` to the running process group.

**Response:**
```json
{ "status": "success", "message": "Process terminated" }
```

---

### `GET /attendance`
Returns the attendance records page (HTML).

---

### `GET /api/attendance`
Returns all attendance records grouped by date.

**Response:**
```json
{
  "2026-03-05": [
    { "name": "Alice_Smith", "time": "09:05:12", "date": "2026-03-05", "status": "VALID" },
    { "name": "Bob_Jones",   "time": "09:07:44", "date": "2026-03-05", "status": "INVALID - SPOOFING DETECTED" }
  ]
}
```

---

## 12. How Attendance Status Is Decided

```
Recognised face detected in frame
        │
        ├─ spoof_detected flag set?  ──► INVALID - SPOOFING DETECTED
        │
        ├─ BLINK_REQUIRED=True AND no blink?  ──► INVALID - No Blink Detected
        │
        ├─ CHALLENGE_ACTIVE=True AND challenge not done AND liveness < 50?
        │                              ──► INVALID - Challenge Not Completed
        │
        ├─ liveness_score < 50?  ──► INVALID - Failed Liveness Check
        │
        └─ all checks passed  ──► VALID
```

Constants controlling this logic (top of `attendance_taker.py`):

| Constant | Default | Description |
|---|---|---|
| `EYE_AR_THRESH` | `0.30` | EAR threshold for blink detection |
| `EYE_AR_CONSEC_FRAMES` | `1` | Frames below threshold to confirm blink |
| `BLINK_REQUIRED` | `False` | Set `True` to require a blink for every VALID mark |
| `MOTION_THRESHOLD` | `0.05` | Minimum average landmark motion to pass |
| `MOTION_FRAMES` | `10` | Frames in the motion history deque |
| `CHALLENGE_ACTIVE` | `True` | Enable/disable challenge-response layer |
| `CHALLENGE_TYPES` | `["BLINK","NOD"]` | Possible challenges (randomly chosen) |
| `CHALLENGE_DURATION` | `50` | Frame budget to complete the challenge |

---

## 13. Troubleshooting

### Camera not opening
- Check no other app (FaceTime, Zoom) holds the camera.
- macOS: System Settings → Privacy & Security → Camera → grant terminal/Python access.
- Linux: ensure your user is in the `video` group (`sudo usermod -aG video $USER`).

### `features_all.csv` not found
Run `python3 features_extraction_to_csv.py` after registering faces.  The file is created in `data/`.

### Database connection refused
```bash
docker compose ps          # verify container is healthy
docker compose up -d       # restart if stopped
# If port 5432 is taken by a local Postgres, change DB_PORT in .env and docker-compose.yml
```

### dlib compilation fails
```bash
# macOS
brew install cmake openblas
# Ubuntu
sudo apt install cmake libopenblas-dev liblapack-dev
```

### Low recognition accuracy
- Register more images per person (≥15, varied lighting and angles).
- Re-run `features_extraction_to_csv.py` after adding images.
- Reduce recognition threshold from `0.4` → `0.35` in `attendance_taker.py` for stricter matching.

### Liveness score stays low / always INVALID
- Ensure adequate lighting (avoid backlighting).
- Complete the on-screen challenge (BLINK or NOD).
- For testing purposes, set `CHALLENGE_ACTIVE = False` temporarily.

### `dlib` import error on Apple Silicon
```bash
pip install dlib --no-binary :all:
# or use Rosetta: arch -x86_64 pip install dlib
```

---

## 14. Security Notes

See [SECURITY.md](SECURITY.md) for the full vulnerability disclosure policy.

- **`.env` is git-ignored** — never commit real credentials.
- The PostgreSQL container is bound to `127.0.0.1:5432` by default — not exposed to the network.
- The Flask server runs in debug mode (`debug=True`) — set `debug=False` and use a production WSGI server (gunicorn) for any non-local deployment.
- Face feature vectors (128-D floats) are stored only locally in `data/features_all.csv`.  No biometric data is transmitted externally.
- Attendance records include name and timestamp only — no raw images are stored in the database.

---

## 15. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues and pull requests.

---

*TruePresence — because attendance should be impossible to fake.*
