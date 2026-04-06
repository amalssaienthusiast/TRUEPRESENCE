# TruePresence — High Level Design (HLD)

> **Version:** 2.1.0 &nbsp;|&nbsp; **Date:** April 2026 &nbsp;|&nbsp; **Author:** TruePresence Engineering

---

## 1. Executive Summary

**TruePresence** is a production-grade, AI-powered face recognition attendance system with multi-layer anti-spoofing protection. It eliminates proxy attendance and photo/video spoofing attacks by combining real-time biometric identification with a 5-layer (built-in) + 3-stage (ML-API) liveness verification pipeline.

The system runs entirely locally — no cloud dependency, no Docker, no external database server — making it suitable for air-gapped classrooms, labs, and small offices.

---

## 2. System Goals & Non-Goals

### Goals
| # | Goal |
|---|---|
| G1 | Accurate face recognition using 128-D dlib ResNet embeddings |
| G2 | Multi-layer anti-spoofing: reject photos, screens, and synthetic faces |
| G3 | Zero-dependency local operation (SQLite, no Docker, no cloud) |
| G4 | Web-based control panel accessible from any device on the local network |
| G5 | Resilient: if one subsystem fails, the rest continue operating |
| G6 | Extensible: optional deep-learning ML API adds YOLO + CNN spoof detection |

### Non-Goals
- Cloud-hosted multi-site deployment
- Mobile app (web UI is mobile-responsive)
- Integration with external HR or LMS systems (export CSV covers this)

---

## 3. High-Level Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │              USER'S MACHINE (Local)             │
                        │                                                  │
                        │  ┌──────────────────┐  ┌──────────────────────┐│
                        │  │  Web Browser     │  │  Mac / Linux Desktop ││
                        │  │  localhost:8000  │  │  Native Qt Windows   ││
                        │  └────────┬─────────┘  └──────────┬───────────┘│
                        │           │                        │            │
                        │  ┌────────▼─────────────────────────────────┐  │
                        │  │         FastAPI Dashboard  (:8000)       │  │
                        │  │   app/main.py                            │  │
                        │  │   • Process launcher (subprocess)        │  │
                        │  │   • REST API: /api/stats, /attendance    │  │
                        │  │   • Jinja2 HTML templates                │  │
                        │  └────────┬────────────────────────────────┘  │
                        │           │ reads/writes                        │
                        │  ┌────────▼───────────────┐                   │
                        │  │   database.py (SQLite)  │                   │
                        │  │   data/attendance.db    │                   │
                        │  └────────────────────────┘                   │
                        │                                                 │
                        │  ┌─────────────────┐  ┌──────────────────────┐│
                        │  │ get_faces_from_ │  │  attendance_taker.py ││
                        │  │  camera.py      │  │  (Face Recog +       ││
                        │  │  (Registration) │  │   Anti-Spoof)        ││
                        │  └─────────────────┘  └──────────────────────┘│
                        │                                                 │
                        │  ┌──────────────────────────────────────────┐  │
                        │  │  Anti-Spoof ML API  (:8001) [OPTIONAL]   │  │
                        │  │  antispoof/api/main.py                   │  │
                        │  │  Stage1: EAR Blink → Stage2: YOLO Screen │  │
                        │  │  → Stage3: MobileNetV3 CNN Classifier    │  │
                        │  └──────────────────────────────────────────┘  │
                        └─────────────────────────────────────────────────┘
```

---

## 4. Component Overview

| Component | Role | Technology |
|---|---|---|
| **FastAPI Dashboard** | Web control panel, REST API, attendance viewer | FastAPI 0.100+, Jinja2, Uvicorn |
| **SQLite Layer** | Persistent attendance storage, zero-config | Python stdlib `sqlite3`, WAL mode |
| **Face Registration** | Capture and store face images | OpenCV, dlib, PyQt6 |
| **Feature Extraction** | Build 128-D face embedding CSV | dlib ResNet, NumPy, Pandas |
| **Attendance Engine** | Real-time recognition + anti-spoofing | dlib, OpenCV, SciPy, PyQt6 |
| **Anti-Spoof ML API** | Deep-learning liveness verification | FastAPI, YOLOv8n, MobileNetV3, ONNX |
| **Frontend** | Dark-mode glassmorphism dashboard | Vanilla CSS, JavaScript, Font Awesome |

---

## 5. Data Flow — End-to-End Workflow

```
Step 1 — REGISTER
  User → clicks "Register New Face" in dashboard
      → dashboard launches get_faces_from_camera.py
      → PyQt6 window opens on desktop
      → camera captures N images per person (N ≥ 10 recommended)
      → saved to data/data_faces_from_camera/<person_id>/

Step 2 — EXTRACT
  Dashboard button (or auto-chain from Step 1)
      → features_extraction_to_csv.py runs
      → dlib ResNet encodes each image → 128-D float vector
      → all embeddings appended to data/features_all.csv

Step 3 — ATTEND
  User → clicks "Start Attendance"
      → dashboard launches attendance_taker.py
      → PyQt6 window opens, camera feed starts
      → per-frame loop:
            detect faces (dlib HOG)
          → landmarks (68-point predictor)
          → anti-spoof checks (5 layers, see LLD)
          → match 128-D embedding vs database (Euclidean < 0.4)
          → if match + LIVE: record_attendance("Name", "VALID")
          → if spoof detected: record_attendance("Name", "INVALID")
      → database.py writes to data/attendance.db (SQLite)

Step 4 — VIEW
  Browser → http://localhost:8000/attendance
      → FastAPI reads SQLite → grouped by date
      → Jinja2 renders dark-mode table
      → User searches/filters/exports CSV
```

---

## 6. System Boundaries & Interfaces

| Interface | Protocol | Format | Direction |
|---|---|---|---|
| Browser ↔ Dashboard | HTTP | HTML / JSON | Bidirectional |
| Dashboard → Scripts | OS subprocess | stdout text | Dashboard reads script stdout |
| Scripts → SQLite | File I/O | SQL | Read + Write |
| attendance_taker → Anti-Spoof API | HTTP (optional) | JPEG + JSON | taker calls API |
| Anti-Spoof API ↔ Models | In-process | NumPy arrays | Direct call |
| Dashboard → Browser | SSE-style polling | JSON (`/status`) | Dashboard pushes via poll |

---

## 7. Resilience Strategy

| Failure | Behaviour |
|---|---|
| SQLite init failure on startup | App logs warning, continues — DB endpoints return empty data |
| Database write fails per record | `record_attendance()` returns `"error"`, logs warning, app keeps running |
| Script subprocess crashes | Dashboard detects exit code, logs in terminal, re-enables buttons |
| Anti-Spoof ML API down | attendance_taker.py falls back to built-in 5-layer checks only |
| Missing model checkpoints | `antispoof/` stages log warnings, pipeline continues with remaining stages |
| dlib landmark file missing | attendance_taker.py logs error on startup, exits gracefully with message |
| Camera unavailable | OpenCV raises exception, caught and displayed in terminal + PyQt6 dialog |

---

## 8. Deployment

### Single-machine (current)
```bash
# Terminal 1 — Web dashboard
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 (optional) — Anti-spoof ML API
cd antispoof
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# No Docker. No PostgreSQL. DB auto-created at data/attendance.db
```

### Environment
```
Python        3.12+
OS            macOS (tested), Linux
Camera        any OpenCV-compatible webcam
Storage       ~50 MB for models + growing attendance.db
RAM           ~400 MB at idle, ~900 MB during YOLO/CNN inference
```

---

## 9. Security Considerations

| Surface | Measure |
|---|---|
| Proxy attendance | 5-layer liveness detection (EAR + Motion + HOG + Challenge + Score) |
| Photo attack | HOG texture variance < threshold → INVALID |
| Screen replay | YOLO screen detector (Stage 2, ML API) |
| Deepfake/synthetic | MobileNetV3 CNN (Stage 3, ML API) |
| Web UI | Localhost-only by default; no auth (add reverse proxy if exposing) |
| Data | SQLite file at `data/attendance.db` — protect with OS file permissions |
