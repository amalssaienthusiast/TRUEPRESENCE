# Antispoof — Face Anti-Spoofing Module

TruePresence anti-spoofing module using a three-stage pipeline to eliminate face spoofing attacks in an AI-based attendance system.

## Pipeline Overview

```
Frame → Stage 1 → Stage 2 → Stage 3 → Verdict
         EAR       YOLO       CNN
         Blink     Screen     Real vs
         Gate      Detector   Fake
```

The system rejects:
- Printed photos held up to camera
- Video replays on phone/tablet screens
- AI-generated / synthetic faces (StyleGAN, Stable Diffusion)
- Deepfake video streams
- Any face appearing on a device screen

**All three stages must pass for attendance to be marked.**

## Dataset Table

| Key | Name | Type | Size | Access | Stage |
|---|---|---|---|---|---|
| lcc_fasd | LCC FASD | Anti-Spoofing | ~1 GB | Free (Kaggle) | Stage 3 |
| celeba_spoof | CelebA-Spoof | Anti-Spoofing | ~50 GB | Auth Required | Stage 3 |
| human_faces | Human Faces Dataset | Real vs AI-Gen | ~2 GB | Free (Kaggle) | Stage 3 (val) |
| fake_140k | 140k Real & Fake | Real vs AI-Gen | ~10 GB | Free (Kaggle) | Stage 3 |
| sfhq | SFHQ Part 1 | Synthetic Faces | ~5 GB | Free (Kaggle) | Stage 3 (fake) |
| mobile_person | Mobile Person | YOLO Detection | ~500 MB | Roboflow API | Stage 2 |
| mobile_phone | Mobile Phone | YOLO Detection | ~1 GB | Roboflow API | Stage 2 |

## Quick Start

```bash
git clone <repo>
cd antispoof
pip install -r requirements.txt

# Download free datasets
python data/download_all.py --datasets lcc_fasd,human_faces

# Check download status
python data/download_all.py --list

# Train Stage 3 classifier
python training/train_classifier.py --dataset combined --epochs 30

# Train Stage 2 phone detector
python training/train_detector.py --epochs 80

# Run liveness API
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

## Kaggle Training Guide

1. Open `notebooks/kaggle_train_classifier.ipynb` on Kaggle
2. Attach datasets: Human Faces + 140k Real and Fake
3. Enable GPU T4 x2 accelerator
4. Run All → checkpoints saved to `/kaggle/working/`

## API Endpoints

```bash
# Verify a frame
curl -X POST http://localhost:8001/api/v1/liveness/verify \
  -F "frame=@/path/to/frame.jpg" \
  -F "session_id=user123"

# Health check
curl http://localhost:8001/api/v1/liveness/health

# Reset session
curl -X POST http://localhost:8001/api/v1/liveness/reset/user123
```

## Performance Targets

| Metric | Target | Notes |
|---|---|---|
| ACER | < 0.05 | Combined test set |
| AUC | > 0.96 | Hard test set (B4) |
| mAP50 | > 0.80 | Screen detection |
| FPS | > 15 | Full pipeline on CPU |

## Troubleshooting

**dlib install fails:**
```bash
brew install cmake                    # macOS
apt install cmake libopenblas-dev     # Ubuntu
pip install dlib --no-cache-dir
```

**CUDA OOM during training:**
```bash
python training/train_classifier.py --batch 32  # reduce batch size
```

**Kaggle dataset not found:**
- Ensure `~/.kaggle/kaggle.json` exists with your API token
- Run: `kaggle datasets download -d <slug> --list` to test connectivity
