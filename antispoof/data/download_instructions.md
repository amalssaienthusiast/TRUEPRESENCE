# Dataset Download Instructions — Manual Steps

Some datasets require authentication or human review before download.
Follow the steps below for each.

---

## LCC FASD (Anti-Spoofing)

1. Visit: https://github.com/kprokofi/light-weight-face-anti-spoofing
2. **OR** search "LCC FASD face anti spoofing" on Kaggle
3. Kaggle CLI (once slug found):
   ```bash
   kaggle datasets download -d <slug> -p ./data --unzip
   ```
4. Expected: `data/LCC_FASD_development/{real,fake}/`

---

## CelebA-Spoof (Anti-Spoofing — Auth Required)

1. Visit: https://github.com/ZhangYuanhan-AI/CelebA-Spoof
2. Click "Dataset Download" → fill the Google Form
3. Receive download link via email
4. Expected: `data/CelebA_Spoof/data/` + `data/CelebA_Spoof/metas/`
- **Size:** ~50 GB — allocate at least 60 GB disk

---

## OULU-NPU (Auth Required)

1. Visit: https://sites.google.com/site/oulunpudatabase/
2. Register and request download link
3. Expected: `data/OULU-NPU/ClientVideos/` + `data/OULU-NPU/ImposterVideos/`

---

## Real vs Fake Faces — Hard Dataset (B4)

1. Visit: https://gts.ai/dataset-download/fake-vs-real-faces-hard/
2. Register to download
3. Expected: `data/hard_faces/{real,fake}/`
- **Use ONLY as test set — never train on this.**

---

## Roboflow Datasets (C1, C2) — Requires API Key

1. Sign up at https://app.roboflow.com
2. Copy your API key from Account Settings
3. Add to `.env`: `ROBOFLOW_API_KEY=your_key_here`
4. Run the programmatic download:
   ```python
   from data.loaders.mobile_screen import MobileScreenDataset
   MobileScreenDataset.roboflow_download(api_key="your_key", output_dir="./data")
   ```
