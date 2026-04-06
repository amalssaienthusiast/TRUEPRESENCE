# TruePresence — Complete System Requirements

> Verified against the running system: **Python 3.12.10, macOS, April 2026**

---

## Hardware Requirements

### Minimum (Core System — Registration + Attendance)

| Component | Minimum | Notes |
|---|---|---|
| **CPU** | Dual-core 2.0 GHz (x86_64 or ARM64) | dlib HOG + ResNet run on CPU |
| **RAM** | 4 GB | ~900 MB used at runtime; OS + browser use the rest |
| **Storage** | 2 GB free | Models ~250 MB + face images + DB |
| **Webcam** | Any USB/built-in camera, 640×480 | Higher resolution = no benefit; OpenCV caps at 640×480 |
| **Display** | Any monitor | Required — PyQt6 opens a native GUI window |
| **OS** | macOS 12+ / Ubuntu 20.04+ / Windows 10+ | Tested on macOS |

### Recommended (Full System including ML Anti-Spoof API)

| Component | Recommended | Why |
|---|---|---|
| **CPU** | 4-core+ 2.5 GHz (Apple M-series, Intel i5+, AMD Ryzen 5+) | Faster ONNX inference, smoother 30 FPS |
| **RAM** | 8 GB | PyTorch model training needs headroom |
| **Storage** | 10 GB free | Training datasets (CelebA-Spoof is ~6 GB) |
| **GPU** | Optional — CUDA 11.8+ or Apple MPS | ~3× faster training only; inference is ONNX CPU |
| **Camera** | 720p or higher | Better face detection under varied lighting |
| **Network** | Not required for runtime | Internet only needed to download datasets/models |

> **Tested config:** Apple M-series Mac, 8 GB RAM — runs at **28–30 FPS** with full 5-layer anti-spoof, no GPU.

---

## Software Requirements

### Operating System

| OS | Support | Notes |
|---|---|---|
| **macOS 12 Monterey+** | ✅ Fully tested | Camera permission via AVFoundation |
| **Ubuntu 20.04 / 22.04** | ✅ Supported | May need `apt install libgtk-3-dev libopenblas-dev` |
| **Windows 10 / 11** | ⚠️ Supported with caveats | `setsid` subprocess call not available; minor tweak needed |
| **Raspberry Pi OS (64-bit)** | ⚠️ Possible | Slow — dlib compiles from source, ~3 FPS |

---

### Python

| Requirement | Version |
|---|---|
| Python | **3.12.x** (tested on 3.12.10) |
| pip | 23.x+ |
| venv | Bundled with Python 3.12 |

> Python 3.9, 3.10, 3.11 should also work but are untested.

---

### System-Level Dependencies (must install before pip)

#### macOS
```bash
# Homebrew (https://brew.sh)
brew install cmake          # required to compile dlib from source
brew install openblas       # BLAS backend for dlib linear algebra
# Camera: macOS will prompt for camera permission on first run — click Allow
```

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev
```

#### Windows
```powershell
# Install CMake from https://cmake.org/download/
# Install Visual Studio Build Tools (C++ workload)
# dlib may alternatively install via pre-built wheel:
# pip install dlib --find-links https://github.com/jloh02/dlib/releases/
```

---

### Core Python Packages (`requirements.txt`)

| Package | Version | Role |
|---|---|---|
| `fastapi` | ≥0.110.0 | Web dashboard framework |
| `uvicorn[standard]` | ≥0.29.0 | ASGI server |
| `jinja2` | ≥3.1.0 | HTML templating |
| `python-multipart` | ≥0.0.9 | Form data parsing (script launch) |
| `dlib` | ≥19.24.6 *(tested: 19.24.6)* | Face detection, landmarks, 128-D embeddings |
| `numpy` | ≥1.26.0, <2.0.0 | Numerical arrays, EAR formula, motion |
| `opencv-python` | ≥4.8.0 *(tested: 4.11.0)* | Camera I/O, HOG, frame drawing |
| `pandas` | ≥2.0.0 | Read/write `features_all.csv` |
| `scikit-image` | ≥0.22.0 | Image preprocessing utilities |
| `scikit-learn` | ≥1.3.0 | Label encoding, metrics |
| `scipy` | ≥1.11.0 | `scipy.spatial.distance.euclidean` for EAR |
| `imutils` | ==0.5.4 | Frame resize helpers |
| `PyQt6` | ≥6.6.0 *(tested: 6.10.2)* | Native GUI windows (registration + attendance) |
| `python-dotenv` | ≥1.0.0 | `.env` configuration loading |

```bash
pip install -r requirements.txt
```

---

### ML Anti-Spoof API Packages (`antispoof/requirements.txt`)

> Only needed if running the optional deep-learning anti-spoof API on port 8001.

| Package | Version | Role |
|---|---|---|
| `torch` | ≥2.1.0 *(tested: 2.11.0)* | CNN training (MobileNetV3) |
| `torchvision` | ≥0.16.0 | Model zoo, image transforms |
| `ultralytics` | ≥8.0.200 | YOLOv8n screen/phone detector |
| `onnx` | ≥1.15.0 | Model export format |
| `onnxruntime` | ≥1.16.0 | Fast CPU inference at runtime |
| `Pillow` | ≥10.0.0 | Dataset image loading |
| `pydantic` | ≥2.4.0 | API schema validation |
| `slowapi` | ≥0.1.9 | Rate limiting |
| `roboflow` | ≥1.1.9 | Screen detector dataset download |
| `gdown` | ≥4.7.3 | Google Drive dataset download |
| `tqdm` | ≥4.66.0 | Training progress bars |
| `tensorboard` | ≥2.14.0 | Training visualisation |
| `seaborn` | ≥0.13.0 | Evaluation plots |
| `matplotlib` | ≥3.8.0 | Evaluation plots |
| `pyyaml` | ≥6.0.1 | YOLO config parsing |

```bash
pip install -r antispoof/requirements.txt
```

---

### Required Model Files

These must be present **before** running the system:

| File | Size | Where to get |
|---|---|---|
| `data/data_dlib/shape_predictor_68_face_landmarks.dat` | ~95 MB | [dlib.net/files](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |
| `data/data_dlib/dlib_face_recognition_resnet_model_v1.dat` | ~21 MB | [dlib.net/files](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2) |
| `antispoof/checkpoints/best_antispoof.onnx` *(optional)* | ~10 MB | Train via `python antispoof/training/train_classifier.py` |
| `antispoof/checkpoints/best_screen_detector.pt` *(optional)* | ~6 MB | Train via `python antispoof/training/train_detector.py` |

```bash
# Download dlib model files
cd data/data_dlib/
curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | bunzip2 > shape_predictor_68_face_landmarks.dat
curl -L http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 | bunzip2 > dlib_face_recognition_resnet_model_v1.dat
```

---

## Complete Setup — Step by Step

```bash
# 1. Clone / navigate to project
cd "Face-Recognition-Based-Attendance-System  "

# 2. Create virtual environment
python3 -m venv env
source env/bin/activate        # macOS/Linux
# env\Scripts\activate         # Windows

# 3. Install system deps first (macOS)
brew install cmake openblas

# 4. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download dlib model files
mkdir -p data/data_dlib
cd data/data_dlib
curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" | bunzip2 > shape_predictor_68_face_landmarks.dat
curl -L "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" | bunzip2 > dlib_face_recognition_resnet_model_v1.dat
cd ../..

# 6. Configure environment
cp .env.example .env
# Edit .env if needed (DB_PATH is auto-set)

# 7. Start the dashboard
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 8. Open browser
open http://localhost:8000
```

---

## Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `data/attendance.db` | SQLite database file location |
| `ANTISPOOF_API_URL` | `http://localhost:8001` | ML anti-spoof API base URL (optional) |
| `ANTISPOOF_CKPT` | `antispoof/checkpoints/best_antispoof.onnx` | CNN classifier checkpoint |
| `SCREEN_CKPT` | `antispoof/checkpoints/best_screen_detector.pt` | YOLO screen detector checkpoint |
| `DLIB_LANDMARKS` | `data/data_dlib/shape_predictor_68_face_landmarks.dat` | dlib landmarks model path |
| `ROBOFLOW_API_KEY` | *(empty)* | Required only for downloading screen detector dataset |

---

## Port Usage

| Port | Service | Required |
|---|---|---|
| `8000` | FastAPI Dashboard (main UI) | ✅ Always |
| `8001` | Anti-Spoof ML API | ⚠️ Optional |

---

## Quick Compatibility Check

Run this to verify your environment is ready:

```bash
source env/bin/activate
python3 -c "
import dlib, cv2, numpy, pandas, scipy, PyQt6
from fastapi import FastAPI
print('dlib      :', dlib.__version__)
print('opencv    :', cv2.__version__)
print('numpy     :', numpy.__version__)
print('pandas    :', pandas.__version__)
print('PyQt6     :', PyQt6.__version__)
faces = dlib.get_frontal_face_detector()
pred  = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
print('dlib models: OK')
print('All requirements satisfied ✓')
"
```

Expected output:
```
dlib      : 19.24.6
opencv    : 4.11.0
numpy     : 1.26.x
pandas    : 2.x.x
PyQt6     : 6.x.x
dlib models: OK
All requirements satisfied ✓
```
