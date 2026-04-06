# TruePresence — Architecture & Technology Reference

> **Version:** 2.1.0 &nbsp;|&nbsp; **Date:** April 2026

---

## 1. Technology Stack

| Layer | Technology | Version | Why |
|---|---|---|---|
| **Web Framework** | FastAPI | 0.100+ | Async, type-safe, auto-docs, lifespan hooks |
| **ASGI Server** | Uvicorn | 0.24+ | High-performance async HTTP |
| **Templating** | Jinja2 | 3.x | Server-side HTML rendering, inherits FastAPI |
| **Database** | SQLite (stdlib) | built-in | Zero-config, zero-dependency, WAL mode |
| **Face Detection** | dlib HOG detector | 19.x | CPU-efficient frontal face detection |
| **Facial Landmarks** | dlib 68-pt predictor | — | EAR blink, nose nod, face alignment |
| **Face Recognition** | dlib ResNet embedding | — | 128-D descriptor, 99.38% LFW accuracy |
| **GUI Framework** | PyQt6 | 6.6+ | Native camera window, cross-platform Qt |
| **Computer Vision** | OpenCV | 4.x | Camera I/O, HOG, Haar cascade, drawing |
| **Numerical** | NumPy | 1.24+ | Embeddings, EAR formula, motion variance |
| **Data I/O** | Pandas | 2.x | Read/write features_all.csv |
| **Spatial Math** | SciPy | 1.11+ | `scipy.spatial.distance.euclidean` for EAR |
| **ML Training** | PyTorch | 2.1+ | CNN + YOLO training, AMP |
| **CNN Architecture** | torchvision MobileNetV3 | — | Lightweight real/fake classifier |
| **Object Detection** | Ultralytics YOLOv8n | 8.x | Screen/phone detector (Stage 2) |
| **Inference Runtime** | ONNX Runtime | 1.16+ | Fast CPU inference, no PyTorch at runtime |
| **Image Processing** | Pillow | 10.x | Dataset loading, transforms |
| **Config Management** | python-dotenv | 1.x | `.env` → `os.environ` |
| **Frontend CSS** | Vanilla CSS | — | Glassmorphism design system, custom tokens |
| **Frontend JS** | Vanilla JS (ES2020) | — | No framework overhead, fetch API |
| **Icons** | Font Awesome 6 | CDN | Action icons |
| **Fonts** | Google Fonts (Inter) | CDN | Premium typography |

---

## 2. Architectural Patterns

### 2.1 — Layered Architecture

```
┌──────────────────────────────────────────┐
│  Presentation Layer                      │
│  HTML/CSS/JS (Dark-mode Glassmorphism)   │
├──────────────────────────────────────────┤
│  API Gateway Layer                       │
│  FastAPI (app/main.py)  — REST Routes    │
├──────────────────────────────────────────┤
│  Service Layer                           │
│  Script launching, process management   │
│  Stats aggregation, CSV streaming       │
├──────────────────────────────────────────┤
│  Domain Layer                            │
│  Face recognition, liveness analysis    │
│  Anti-spoof pipeline orchestrator       │
├──────────────────────────────────────────┤
│  Data Access Layer                       │
│  database.py — SQLite CRUD + stats      │
├──────────────────────────────────────────┤
│  Infrastructure                          │
│  SQLite file, dlib model files          │
│  Camera (OpenCV/AVFoundation), ONNX     │
└──────────────────────────────────────────┘
```

### 2.2 — Pipeline Pattern (Anti-Spoofing)

The anti-spoof system uses a **short-circuit sequential pipeline** — each stage either passes the frame downstream or immediately returns `SPOOF`. This minimises CPU usage: Stage 3 (CNN) is only reached if Stages 1 and 2 pass.

```
frame → [Gate 1] → [Gate 2] → [Gate 3] → LIVE
          │            │            │
        SPOOF        SPOOF        SPOOF
```

Each gate is an independent class implementing `check_*(frame) -> dict`:
- Returns `{"live": None}` = PENDING
- Returns `{"live": False, ...}` = SPOOF
- Returns `{"live": True, ...}` = PASS

### 2.3 — Strategy Pattern (Liveness Scoring)

The 5-layer built-in anti-spoof in `attendance_taker.py` applies a **weighted composite strategy**:

```python
score = (
    EAR_blink_ok   * 30 +   # blink detected
    motion_ok      * 30 +   # landmark variance > threshold
    texture_ok     * 20 +   # HOG variance > threshold
    challenge_done * 20     # blink or nod completed
)
# Smoothed via exponential moving average per person
# score ≥ 50 → VALID attendance
```

Each sub-check is independently pluggable — change thresholds or weights without touching other layers.

### 2.4 — Observer / Polling Pattern (Dashboard ↔ Scripts)

The web frontend uses **client-side polling** (1.2s interval) against `/status` to observe subprocess state. This avoids WebSocket complexity while keeping the UI responsive.

```
Browser                  FastAPI
  │                         │
  ├─ POST /run_script ──────► spawn subprocess in background thread
  │                         │ monitor thread reads stdout → _output_lines[]
  ├─ GET /status (1.2s) ───► return {running, output, exit_code}
  ├─ GET /status ──────────► ...
  ├─ GET /status ──────────► running=false → stop polling
  │
  └─ refresh stats, re-enable buttons
```

### 2.5 — Repository Pattern (Database Layer)

`database.py` acts as a **repository** — the rest of the system never touches SQLite directly. All SQL is encapsulated; callers use semantic function names (`record_attendance`, `get_stats`). Swapping the underlying store (e.g., to PostgreSQL) requires only changing `database.py`.

### 2.6 — Centroid Tracker (Attendance Engine)

```
Problem: dlib recognition (128-D embedding) is expensive at 30fps
Solution: only re-classify every 10 frames; interim frames use centroid tracking

Frame 0,10,20...: full dlib recognition → assign name to face
Frames 1-9,11-19: compute face centroid, match to nearest last-frame centroid → keep name
```

This reduces recognition calls by ~90% while maintaining identity continuity.

### 2.7 — Session Store Pattern (ML API)

The anti-spoof API must track blink counts **across multiple frames** from the same person. It uses an in-memory dict keyed by `session_id` with a 30s TTL:

```python
_sessions = {
    "session_abc123": {
        "gate":        EyeBlinkLivenessGate(...),  # stateful blink counter
        "last_access": 1712345678.0
    }
}
# Expired sessions cleaned on each request
```

---

## 3. Data Models

### 3.1 — Attendance Record
```python
{
    "id":     int,           # auto-increment
    "name":   str,           # "Amal_S.S"
    "time":   str,           # "09:17:46"
    "date":   str,           # "2026-04-06"
    "status": "VALID" | "INVALID" | "UNKNOWN"
}
```

### 3.2 — Pipeline Result (ML API)
```python
@dataclass
class PipelineResult:
    verdict:       "LIVE" | "SPOOF" | "PENDING" | "NO_FACE"
    confidence:    float          # 0.0-1.0 live probability (Stage 3)
    latency_ms:    float          # wall-clock pipeline time
    spoof_reason:  str | None     # "no_blink_detected" | "mobile_screen_in_frame" | "synthetic_face_detected"
    stage_results: dict           # per-stage raw output
```

### 3.3 — Face Feature Database (CSV)
```
Row format: [name, f1, f2, ..., f128]
name: "Person_ID"   (underscore-joined for CSV safety)
f1..f128: 128 floats from dlib ResNet descriptor
One row per registered person (averaged over N images)
```

### 3.4 — Stats Response
```python
{
    "total":   int,      # all-time records
    "today":   int,      # records dated today
    "valid":   int,      # status = "VALID"
    "invalid": int,      # status = "INVALID"
    "db_ok":   bool      # SQLite reachable
}
```

---

## 4. Innovation Highlights

### 4.1 — Multi-Layer Defence-in-Depth Anti-Spoofing

Most attendance systems rely on a single biometric. TruePresence applies **8 independent anti-spoofing signals** across two subsystems:

| Signal | Subsystem | Attack stopped |
|---|---|---|
| Eye Aspect Ratio blink | Built-in | Printed photo, statue |
| Landmark motion variance | Built-in | Static screen replay |
| HOG texture variance | Built-in | Low-detail printed photo |
| Challenge-response (blink/nod) | Built-in | Pre-recorded video |
| Composite liveness score (≥50) | Built-in | Partial spoof combinations |
| EAR blink gate (session-aware) | ML API | Photo/video with open eyes |
| YOLOv8n screen detector | ML API | Phone/tablet replay attacks |
| MobileNetV3 CNN classifier | ML API | Deepfake, synthetic face |

**Result:** A spoofed face must defeat all active checks simultaneously — which your own screenshots prove: holding a phone photo scored INVALID even though the face was clearly recognisable.

### 4.2 — Zero-Infrastructure Local AI

- No GPU required (runs on CPU at 28-30 FPS as seen in screenshots)
- No cloud API key (all inference local)
- No Docker, no PostgreSQL, no external database server
- SQLite auto-created — works out of the box

### 4.3 — Auto-Chain Workflow

Step 1 (Register) automatically launches Step 2 (Extract Features) when the registration script exits, via the `auto_next` parameter in the script launcher. Users never need to manually trigger feature extraction after registration.

### 4.4 — Session-Aware Blink Tracking (ML API)

The ML API's Stage 1 gate is **stateful per-session** — it remembers how many blinks have occurred across multiple frame submissions from the same person. This is essential: a single frame cannot determine liveness. The pipeline correctly returns `PENDING` until the time window elapses or enough blinks are counted.

### 4.5 — Training with Class Balance (WeightedRandomSampler)

Real-world liveness datasets are heavily imbalanced (many more "live" samples). The training pipeline uses PyTorch `WeightedRandomSampler` to ensure each batch has roughly equal real/fake representation without duplicating data, yielding better generalisation.

---

## 5. System Sequence Diagram — Attendance Flow

```
User        Browser       Dashboard(FastAPI)    attendance_taker     database.py
 │              │               │                      │                  │
 │─ click ─────►│               │                      │                  │
 │              │─POST /run_script─►                   │                  │
 │              │               │─spawn subprocess─────►                  │
 │              │               │               ┌──────▼──────┐           │
 │              │               │               │ camera open │           │
 │              │               │               │ dlib load   │           │
 │              │               │               └──────┬──────┘           │
 │              │               │  face detected       │                  │
 │              │               │  anti-spoof ok       │                  │
 │              │               │                      │─record_attendance►│
 │              │               │                      │                  │─SQL INSERT
 │              │               │                      │                  │
 │              │─GET /status ──►                      │                  │
 │              │      {running: │                      │                  │
 │              │       output}◄─┤                      │                  │
 │              │               │                      │                  │
 │              │─GET /api/stats►                      │                  │
 │              │      {total,  │                      │                  │
 │              │       today}◄─┤─── get_stats() ──────────────────────►  │
 │              │               │ ◄──────────────────────────────────────  │
```

---

## 6. Design Decisions & Trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| SQLite over PostgreSQL | Zero setup, no Docker, fully portable | Not suitable for multi-server horizontal scaling |
| subprocess over in-process thread | PyQt6 and OpenCV camera need the main thread; avoids GIL and Qt thread conflicts | Cannot capture all script state; output via stdout only |
| Polling over WebSocket | Simpler, zero additional server config, works through any proxy | 1.2s latency in status updates |
| dlib over face_recognition library | Lower-level control, smaller dependency | More verbose code, no GPU acceleration |
| Vanilla JS/CSS over React/Tailwind | Zero build step, no Node.js dependency, instant edits | Less component reuse |
| ONNX for Stage 3 | ~3× faster inference than PyTorch on CPU, no torch at runtime | Requires export step during training |
| Composite score over hard rules | More robust to noise and lighting variation | Harder to explain to end user why INVALID |
