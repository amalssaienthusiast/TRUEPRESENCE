# TruePresence — Low Level Design (LLD)

> **Version:** 2.1.0 &nbsp;|&nbsp; **Date:** April 2026

---

## 1. Module Map

```
Face-Recognition-Based-Attendance-System/
│
├── app/                          # FastAPI web dashboard
│   ├── main.py                   # App entrypoint, routes, subprocess launcher
│   ├── templates/
│   │   ├── index.html            # Control panel (Jinja2)
│   │   └── attendance.html       # Records viewer (Jinja2)
│   └── static/
│       ├── css/styles.css        # Dark glassmorphism design system
│       └── js/main.js            # Dashboard logic, polling, toasts
│
├── database.py                   # SQLite attendance data layer
│
├── get_faces_from_camera.py      # Face registration (PyQt6 + dlib + OpenCV)
├── features_extraction_to_csv.py # 128-D embedding builder (dlib ResNet)
├── attendance_taker.py           # Face recognition + 5-layer anti-spoof (PyQt6)
│
├── antispoof/                    # Optional deep-learning anti-spoof module
│   ├── api/
│   │   ├── main.py               # FastAPI liveness API (:8001)
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── middleware.py         # CORS + request logging
│   ├── pipeline/
│   │   ├── antispoof_pipeline.py # 3-stage orchestrator
│   │   ├── ear_gate.py           # Stage 1: dlib EAR blink gate
│   │   ├── screen_gate.py        # Stage 2: YOLOv8n screen/phone detector
│   │   └── spoof_gate.py         # Stage 3: MobileNetV3 CNN gate
│   ├── models/
│   │   ├── antispoof_net.py      # MobileNetV3-Small model definition
│   │   ├── screen_detector.py    # YOLOv8n wrapper + dataset prep
│   │   └── losses.py             # FocalLoss + LabelSmoothingCE
│   ├── training/
│   │   ├── config.py             # Dataclass configs (EARConfig, TrainConfig, PipelineConfig)
│   │   ├── train_classifier.py   # CNN training loop (AMP, WeightedRandomSampler)
│   │   ├── train_detector.py     # YOLO fine-tuning pipeline
│   │   └── evaluate.py           # Metrics: AUC, FPR, FNR, confusion matrix
│   └── data/
│       ├── download_all.py       # Dataset download orchestrator
│       └── loaders/              # Per-dataset DataLoader classes
│           ├── celeba_spoof.py   # CelebA-Spoof (625K images)
│           ├── lcc_fasd.py       # LCC-FASD (fake face dataset)
│           ├── fake_140k.py      # 140K Real vs Fake faces (Kaggle)
│           ├── human_faces.py    # Real human faces dataset
│           ├── sfhq.py           # SFHQ synthetic dataset
│           ├── mobile_screen.py  # Phone/tablet screen dataset (Roboflow)
│           └── combined.py       # MultiDataset union sampler
│
├── data/                         # Runtime data (auto-created)
│   ├── attendance.db             # SQLite database
│   ├── features_all.csv          # 128-D face embeddings
│   ├── data_faces_from_camera/   # Raw face images
│   └── data_dlib/                # dlib model files
│       ├── shape_predictor_68_face_landmarks.dat
│       └── dlib_face_recognition_resnet_model_v1.dat
│
├── requirements.txt              # Main project deps
└── .env.example                  # Environment variable template
```

---

## 2. Database Layer — `database.py`

### Schema
```sql
CREATE TABLE IF NOT EXISTS attendance (
    id      INTEGER  PRIMARY KEY AUTOINCREMENT,
    name    TEXT     NOT NULL,
    time    TEXT     NOT NULL,         -- "HH:MM:SS"
    date    TEXT     NOT NULL,         -- "YYYY-MM-DD"
    status  TEXT     NOT NULL DEFAULT 'UNKNOWN',  -- VALID | INVALID
    UNIQUE(name, date)                 -- one record per person per day
);
CREATE INDEX idx_attendance_date ON attendance(date DESC);
```

### Public API
```python
init_db()           -> bool           # idempotent schema creation
record_attendance(name, status) -> str # "inserted"|"updated"|"exists"|"error"
get_all_records()   -> list[dict]     # newest first
get_records_by_date() -> dict[str, list[dict]]
get_stats()         -> dict           # total, today, valid, invalid, db_ok
export_csv_rows()   -> list[list]     # header + data for CSV download
```

### Thread Safety
```
_write_lock = threading.Lock()
# All INSERT/UPDATE wrapped in: with _write_lock, _get_conn() as conn:
# Reads do NOT acquire write lock (WAL allows concurrent reads)
# check_same_thread=False + WAL journal = safe multi-threaded use
```

### Upgrade Logic (record_attendance)
```
does (name, date) row exist?
├── No  → INSERT → return "inserted"
└── Yes → is existing_status VALID?
          ├── Yes → return "exists"   (no downgrade)
          └── No + new is VALID → UPDATE → return "updated"
```

---

## 3. FastAPI Dashboard — `app/main.py`

### Route Table
| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/` | HTML | Dashboard; passes `stats` context to Jinja2 |
| GET | `/attendance` | HTML | Records grouped by date |
| POST | `/run_script` | JSON | Launches subprocess by logical name |
| GET | `/status` | JSON | `{running, output, script, exit_code}` |
| POST | `/stop_script` | JSON | SIGTERM to process group |
| GET | `/api/attendance` | JSON | All records by date |
| GET | `/api/stats` | JSON | `{total, today, valid, invalid, db_ok}` |
| GET | `/api/attendance/export` | CSV | StreamingResponse download |
| GET | `/health` | JSON | Always 200, reports db_ok |

### Script Launcher
```python
SCRIPT_MAP = {
    "get_faces":       ("get_faces_from_camera.py",     auto_next=True,  next="features_extraction_to_csv.py"),
    "extract_features":("features_extraction_to_csv.py", auto_next=False, next=None),
    "attendance":      ("attendance_taker.py",           auto_next=False, next=None),
}
```
- Each script runs as `subprocess.Popen([sys.executable, script_path])`
- Uses `os.setsid()` so SIGTERM kills the entire process group
- Stdout captured line-by-line in a daemon thread → `_output_lines[]`
- Buffer capped at 300 lines (oldest evicted first)

---

## 4. Face Registration — `get_faces_from_camera.py`

### Flow
```
User enters: ID (int) + Name (str)
    → createdir: data/data_faces_from_camera/<ID>_<Name>/
    → QTimer(33ms) fires _process_frame():
          cap.read() → BGR frame
          → dlib detector → face bounding boxes
          → if exactly 1 face in frame:
                draw green guide rectangle
                show EAR / face count overlay
          → Space key pressed (or button):
                save frame as img_face_{ss_cnt}.jpg
                ss_cnt++
                log to UI textarea
    → User captures ≥ 10 images, closes window
    → dashboard auto-chains to feature extraction
```

### Key Parameters
```python
PATH_PHOTOS   = "data/data_faces_from_camera/"
Cap resolution = 640 × 480
Timer interval = 33 ms (~30 fps)
dlib detector  = get_frontal_face_detector()  # HOG-based
```

---

## 5. Feature Extraction — `features_extraction_to_csv.py`

```
For each person folder in data/data_faces_from_camera/:
    For each image file:
        dlib detect face → largest bounding box
        → predictor(image, bbox) → 68-point landmarks
        → face_reco_model.compute_face_descriptor(image, landmarks)
        → 128-D float vector
    average all vectors for this person → single 128-D embedding
    append [name, f1, f2, ..., f128] to features_all.csv
```

**Output format (features_all.csv):**
```
Amal_S.S, 0.123, -0.456, ..., 0.789   ← 1 name + 128 floats
Priya_R,  0.234, -0.123, ..., 0.456
```

---

## 6. Attendance Engine — `attendance_taker.py`

### Per-Frame Processing Loop (30 fps target)
```
cap.read() → BGR frame (640×480)
    │
    ├─► dlib detector → face bounding boxes (HOG)
    │
    ├─► for each face:
    │       predictor(frame, bbox) → 68-point shape
    │       face_reco_model.compute_face_descriptor() → 128-D embedding
    │       match vs features_all.csv:
    │           Euclidean distance to all known embeddings
    │           min_dist < 0.4 → match found → name
    │           else → "Unknown"
    │
    ├─► Anti-Spoofing (5 layers, in order):
    │       Layer 1 – EAR Blink Detection
    │           ear = (dist(p1,p5) + dist(p2,p4)) / (2 * dist(p0,p3))
    │           ear < 0.30 for ≥1 consecutive frames → blink++
    │       Layer 2 – Motion Analysis
    │           track 68 landmarks over 10 frames
    │           variance of positions > 0.05 → motion detected
    │       Layer 3 – Texture Analysis (HOG)
    │           face ROI → resize 64×64
    │           HOGDescriptor.compute() → 128-D
    │           np.var(descriptor) * 1000 > 0.5 → texture ok
    │       Layer 4 – Challenge-Response
    │           random challenge = BLINK or NOD
    │           must be completed within CHALLENGE_DURATION=50 frames
    │       Layer 5 – Composite Liveness Score
    │           score = smooth(EAR_ok*30 + motion_ok*30 + texture_ok*20 + challenge_ok*20)
    │           score ≥ 50 → VALID, else INVALID
    │
    ├─► Attendance decision:
    │       if matched_name + liveness ≥ 50:
    │           database.record_attendance(name, "VALID")
    │       elif matched_name + spoof:
    │           database.record_attendance(name, "INVALID")
    │
    └─► PyQt6 frame render (overlay: name, EAR, MAR, motion, score, blink count)
```

### Centroid Tracker
```
To avoid re-running expensive dlib recognition every frame:
  reclassify every 10 frames (reclassify_interval=10)
  interim frames: track face position → nearest centroid → keep last name
```

---

## 7. Anti-Spoof ML API — `antispoof/`

### 3-Stage Pipeline (`antispoof_pipeline.py`)

```
Input: BGR frame (numpy array, H×W×3)
    │
    ├─[ Face Detection ] OpenCV Haar cascade → largest (x1,y1,x2,y2)
    │   └─ NO_FACE → return immediately
    │
    ├─[ Stage 1: EAR Gate ]  ear_gate.py
    │   dlib 68-pt landmarks → EAR formula
    │   accumulates blinks per session (30s TTL)
    │   ├─ PENDING (still counting) → return verdict="PENDING"
    │   └─ FAIL (no blinks in window) → return verdict="SPOOF", reason="no_blink_detected"
    │
    ├─[ Stage 2: Screen Detector ]  screen_gate.py
    │   YOLOv8n fine-tuned on mobile_screen dataset
    │   IoU overlap of detected screen bbox with face bbox
    │   └─ FAIL → return verdict="SPOOF", reason="mobile_screen_in_frame"
    │
    └─[ Stage 3: CNN Classifier ]  spoof_gate.py
        face_crop[y1:y2, x1:x2] → resize 224×224
        ONNX runtime (or PyTorch fallback)
        live_prob = sigmoid(logit)[live_class]
        └─ live_prob < 0.52 → return verdict="SPOOF", reason="synthetic_face_detected"
        └─ PASS → verdict="LIVE", confidence=live_prob
```

### Session Management
```python
_sessions: Dict[str, {gate: EyeBlinkLivenessGate, last_access: float}]
_SESSION_TTL = 30s  # per-person blink state expires after 30s of inactivity
```

### Stage 1 — EAR Gate (`ear_gate.py`)
```python
class EyeBlinkLivenessGate:
    ear_threshold       = 0.22    # blink detected when EAR drops below
    blink_consec_frames = 2       # must stay below for 2+ consecutive frames
    required_blinks     = 2       # must complete 2 blinks to pass
    time_window_sec     = 10.0    # must complete within 10 seconds
```

### Stage 2 — Screen Gate (`screen_gate.py`)
```python
class ScreenSpoofGate:
    model = YOLOv8n fine-tuned on mobile_screen.py dataset (Roboflow)
    conf_threshold  = 0.45
    iou_threshold   = 0.35   # IoU of screen bbox vs face bbox
    # if a detected screen overlaps the face region → SPOOF
```

### Stage 3 — Spoof Gate (`spoof_gate.py`)
```python
class SpoofClassifierGate:
    # Loads ONNX model (best_antispoof.onnx) via onnxruntime
    # Fallback: loads .pt weights via PyTorch if ONNX unavailable
    # Preprocessing: resize 224×224, ImageNet normalize, CHW, batch dim
    live_threshold = 0.52     # live_prob must exceed to pass
```

---

## 8. ML Model Details

### Face Embedding Model
| Property | Value |
|---|---|
| Library | dlib |
| Architecture | ResNet-based (proprietary dlib) |
| Output | 128-dimensional float vector |
| Matching | Euclidean distance < 0.4 |
| Source | `dlib_face_recognition_resnet_model_v1.dat` |

### Anti-Spoof Classifier (Stage 3)
| Property | Value |
|---|---|
| Base Architecture | MobileNetV3-Small |
| Input | 224×224 RGB |
| Output | 2-class: [spoof, live] |
| Loss | FocalLoss (γ=2) + LabelSmoothing (ε=0.1) |
| Training | AMP (fp16), WeightedRandomSampler, AdamW |
| Export | ONNX (opset=11, dynamic batch) |

### Screen Detector (Stage 2)
| Property | Value |
|---|---|
| Base Architecture | YOLOv8n (Ultralytics) |
| Classes | mobile_screen |
| Training data | Roboflow mobile screen dataset |
| Input | BGR frame (any resolution) |
| Training | `train_detector.py` via Ultralytics API |

### Landmark Predictor (Stage 1 + attendance_taker)
| Property | Value |
|---|---|
| Library | dlib |
| Output | 68 facial landmark points |
| Use | EAR blink detection, face alignment |
| Source | `shape_predictor_68_face_landmarks.dat` |

---

## 9. Training Pipeline

### Train the CNN Anti-Spoof Classifier
```bash
cd antispoof

# 1. Download datasets
python data/download_all.py  # downloads CelebA-Spoof, LCC-FASD, Fake140K, SFHQ, etc.

# 2. Train
python training/train_classifier.py \
  --epochs 30 --batch 64 --lr 1e-3 \
  --data-dir data/datasets \
  --out-dir checkpoints/

# 3. Evaluate
python training/evaluate.py --ckpt checkpoints/best_antispoof.onnx

# 4. Start API
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### Train the Screen Detector
```bash
python training/train_detector.py
# Downloads Roboflow dataset (ROBOFLOW_API_KEY in .env)
# Trains YOLOv8n for N epochs
# Saves to antispoof/checkpoints/best_screen_detector.pt
```

---

## 10. Configuration Reference

### `.env` (project root)
```ini
DB_PATH=data/attendance.db            # SQLite file path
ANTISPOOF_API_URL=http://localhost:8001  # optional ML API
ROBOFLOW_API_KEY=...                   # for downloading screen dataset
ANTISPOOF_CKPT=antispoof/checkpoints/best_antispoof.onnx
SCREEN_CKPT=antispoof/checkpoints/best_screen_detector.pt
DLIB_LANDMARKS=data/data_dlib/shape_predictor_68_face_landmarks.dat
```

### `antispoof/.env` (ML API)
```ini
DLIB_LANDMARKS=../data/data_dlib/shape_predictor_68_face_landmarks.dat
SCREEN_CKPT=checkpoints/best_screen_detector.pt
ANTISPOOF_CKPT=checkpoints/best_antispoof.onnx
SESSION_TTL_SECONDS=30
```

### Anti-Spoof Constants (`attendance_taker.py`, lines 67-75)
```python
EYE_AR_THRESH        = 0.30    # lower = more sensitive blink detection
EYE_AR_CONSEC_FRAMES = 1       # frames below threshold to count as blink
BLINK_REQUIRED       = False   # True = block VALID until blink confirmed
MOTION_THRESHOLD     = 0.05    # landmark variance threshold
CHALLENGE_ACTIVE     = True    # enable blink/nod challenges
CHALLENGE_TYPES      = ["BLINK", "NOD"]
CHALLENGE_DURATION   = 50      # frames to complete challenge
```
