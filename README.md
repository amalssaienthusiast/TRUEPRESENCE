# TruePresence — Face Recognition Attendance System

**Production-grade AI-powered attendance with multi-layer anti-spoofing.**  
Zero Docker. Zero PostgreSQL. Runs entirely on your local machine.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the dashboard (creates data/attendance.db automatically)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Open in browser
open http://localhost:8000
```

---

## Workflow (4 Steps)

| Step | What to do | How |
|---|---|---|
| **1** | Register faces | Click "Register New Face" in dashboard → camera window opens |
| **2** | Extract features | Auto-runs after Step 1 (or manual button) |
| **3** | Take attendance | Click "Start Attendance" → camera window opens, recognition starts |
| **4** | View records | Click "View All Records" or `http://localhost:8000/attendance` |

---

## Optional: Anti-Spoof ML API

The built-in 5-layer anti-spoof runs automatically (no setup needed).  
For the additional YOLO + CNN deep-learning layer, run:

```bash
# First: train models (or download checkpoints)
cd antispoof
python training/train_classifier.py   # CNN real/fake classifier
python training/train_detector.py     # YOLO screen detector

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
open http://localhost:8001/docs
```

---

## Documentation

| Document | Description |
|---|---|
| [HLD.md](docs/HLD.md) | High Level Design — architecture, data flow, resilience, deployment |
| [LLD.md](docs/LLD.md) | Low Level Design — module map, class internals, DB schema, pipeline detail |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technology stack, design patterns, innovations, data models, trade-offs |

---

## Anti-Spoofing Layers

The system applies **8 independent signals** to reject spoofing attacks:

**Built-in (always active):**
1. Eye Aspect Ratio (EAR) blink detection
2. Facial landmark motion variance
3. HOG texture analysis
4. Challenge-response (blink / nod)
5. Composite liveness score ≥ 50

**ML API (optional, port 8001):**
6. EAR gate with session-aware blink counting
7. YOLOv8n phone/screen detector
8. MobileNetV3-Small real-vs-fake CNN (ONNX)

---

## Requirements

```
Python 3.12+   macOS / Linux
Camera         Any OpenCV-compatible webcam
RAM            ~400 MB idle, ~900 MB with ML API
GPU            Not required (CPU inference, 28-30 FPS tested)
```

---

## Project Structure

```
├── app/                  # FastAPI dashboard (web UI)
├── antispoof/            # Optional deep-learning anti-spoof module
├── data/                 # Auto-created: attendance.db, face images, model files
├── docs/                 # HLD, LLD, ARCHITECTURE documentation
├── database.py           # SQLite data layer
├── get_faces_from_camera.py     # Step 1: face registration
├── features_extraction_to_csv.py # Step 2: feature extraction
├── attendance_taker.py          # Step 3: recognition + anti-spoof
├── requirements.txt
└── .env.example
```
