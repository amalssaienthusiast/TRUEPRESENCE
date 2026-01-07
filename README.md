# TruePresence - Face Recognition Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.2.3-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7-red.svg)
![dlib](https://img.shields.io/badge/dlib-19.24-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A Smart, Secure, and Anti-Spoofing Face Recognition Attendance System**

*Leveraging advanced computer vision with eye-blink detection, facial texture analysis, and motion tracking to prevent proxy attendance.*

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Anti-Spoofing Technology](#-anti-spoofing-technology)
- [API Reference](#-api-reference)
- [Database Schema](#-database-schema)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**TruePresence** is a comprehensive face recognition-based attendance management system designed to eliminate proxy attendance through advanced anti-spoofing mechanisms. The system combines cutting-edge computer vision algorithms with a modern web interface to provide a seamless and secure attendance tracking experience.

### Key Highlights

- 🔒 **Multi-Layer Anti-Spoofing**: Prevents photo, video, and mask-based spoofing attacks
- 🎥 **Real-Time Processing**: Instant face detection and recognition with live camera feed
- 🌐 **Modern Web Interface**: Responsive Flask-based dashboard for easy management
- 📊 **Comprehensive Analytics**: Track attendance patterns with date-wise filtering
- 🔄 **Automated Workflow**: Seamless integration between registration, feature extraction, and attendance
- 💾 **Persistent Storage**: SQLite database for reliable data management

---

## ✨ Features

### Core Functionality

| Feature | Description |
|---------|-------------|
| **Face Registration** | Capture multiple face images from different angles for accurate recognition |
| **Feature Extraction** | Extract 128-dimensional face descriptors using dlib's ResNet model |
| **Real-Time Recognition** | Identify registered individuals with high accuracy |
| **Attendance Logging** | Automatic timestamped attendance records with validation status |

### Anti-Spoofing Mechanisms

| Method | Description |
|--------|-------------|
| **Eye Blink Detection** | Monitors Eye Aspect Ratio (EAR) to detect natural blinking patterns |
| **Texture Analysis** | Uses Local Binary Pattern (LBP) to differentiate real skin from printed photos |
| **Motion Analysis** | Tracks facial landmark movements to identify unnatural/synthetic patterns |
| **Challenge-Response** | Prompts users to perform actions (blink, nod) for verification |

### Web Interface Features

- 📱 Responsive dashboard accessible from any device
- 🔄 Real-time process monitoring with console output
- 📅 Date-wise attendance grouping and filtering
- 🔍 Search and filter attendance records
- 📥 Export attendance data to CSV format

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TruePresence Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Flask Web Interface                         │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐    │    │
│  │  │  Dashboard    │  │  Attendance   │  │  Process Control  │    │    │
│  │  │  (index.html) │  │  (records)    │  │  (AJAX API)       │    │    │
│  │  └───────────────┘  └───────────────┘  └───────────────────┘    │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│  ┌──────────────────────────────▼──────────────────────────────────┐    │
│  │                     Core Python Modules                          │    │
│  │  ┌────────────────┐ ┌────────────────┐ ┌─────────────────────┐  │    │
│  │  │ Face Register  │ │ Feature Extract│ │ Attendance Taker    │  │    │
│  │  │ (Tkinter GUI)  │ │ (128D vectors) │ │ (Anti-Spoofing)     │  │    │
│  │  └────────────────┘ └────────────────┘ └─────────────────────┘  │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│  ┌──────────────────────────────▼──────────────────────────────────┐    │
│  │                       Data Layer                                 │    │
│  │  ┌────────────────┐ ┌────────────────┐ ┌─────────────────────┐  │    │
│  │  │ SQLite DB      │ │ CSV Features   │ │ Face Images         │  │    │
│  │  │ (attendance)   │ │ (128D vectors) │ │ (data directory)    │  │    │
│  │  └────────────────┘ └────────────────┘ └─────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    External Libraries                            │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │    │
│  │  │  dlib   │  │ OpenCV  │  │  NumPy  │  │  SciPy  │            │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technology Stack

### Backend & Processing

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **dlib** | 19.24.0 | Face detection & recognition models |
| **OpenCV** | 4.7.0 | Image processing & camera handling |
| **NumPy** | 1.24.2 | Numerical computations |
| **SciPy** | 1.10.1 | Scientific computing utilities |
| **pandas** | 1.5.3 | Data manipulation & CSV handling |
| **Flask** | 2.2.3 | Web framework |
| **SQLite3** | Built-in | Database management |

### Frontend

| Technology | Purpose |
|------------|---------|
| **HTML5/CSS3** | Responsive layout & styling |
| **JavaScript** | Dynamic UI interactions |
| **Font Awesome** | Icon library |
| **Google Fonts** | Typography (Roboto) |

### Machine Learning Models (dlib)

| Model | Description |
|-------|-------------|
| `shape_predictor_68_face_landmarks.dat` | 68-point facial landmark detector |
| `dlib_face_recognition_resnet_model_v1.dat` | 128-dimensional face encoding (ResNet) |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Webcam or camera device
- CMake (for dlib compilation)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amalssaienthusiast/True_Presence.git
   cd True_Presence
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install Flask app specific requirements:
   ```bash
   pip install -r flask_app/requirements.txt
   ```

4. **Download dlib models** (if not included)
   
   Place the following files in `data/data_dlib/`:
   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`

### Dependency Installation Notes

<details>
<summary><b>Installing dlib on different platforms</b></summary>

**macOS:**
```bash
brew install cmake
pip install dlib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake libboost-all-dev
pip install dlib
```

**Windows:**
- Install Visual Studio Build Tools
- Install CMake
- Run: `pip install dlib`

</details>

---

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

1. **Start the Flask server:**
   ```bash
   ./run_attendance_system.sh
   ```
   
   Or manually:
   ```bash
   cd flask_app
   python3 app.py
   ```

2. **Open your browser:**
   ```
   http://127.0.0.1:5000/
   ```

3. **Use the dashboard** to register faces, extract features, and take attendance.

### Option 2: Command Line Scripts

```bash
# Step 1: Register a new face
python3 get_faces_from_camera.py

# Step 2: Extract facial features
python3 features_extraction_to_csv.py

# Step 3: Take attendance
python3 attendance_taker.py
```

---

## 📖 Usage Guide

### 1. Face Registration

1. Click **"Register New Face"** on the web dashboard
2. Enter the person's **ID** (numeric) and **Name**
3. Click **"Input"** to save the person's info
4. Position the face in front of the camera
5. Click **"Save Current Face"** multiple times (5-10 captures recommended)
6. Click **"Quit"** when done

**Best Practices:**
- Capture faces with different expressions
- Include slight head rotations
- Ensure good lighting conditions
- Avoid glasses/hats during initial registration

### 2. Feature Extraction

Feature extraction automatically runs after registration, or you can trigger it manually:

```bash
python3 features_extraction_to_csv.py
```

This creates/updates `data/features_all.csv` with 128-dimensional face descriptors.

### 3. Taking Attendance

1. Click **"Start Attendance"** on the dashboard
2. Position face in front of the camera
3. Complete the challenge-response verification:
   - **BLINK**: Blink your eyes naturally
   - **NOD**: Nod your head slightly
4. Valid attendance is recorded to the database

### 4. Viewing Attendance Records

- Navigate to **"View Attendance Records"**
- Use filters to search by name, date, or status
- Export filtered data as CSV

---

## 🛡 Anti-Spoofing Technology

TruePresence implements a multi-layered anti-spoofing system:

### Eye Blink Detection

```python
# Eye Aspect Ratio (EAR) algorithm
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

- Monitors EAR across consecutive frames
- Detects natural blinking patterns
- Threshold: EAR < 0.25 indicates closed eyes

### Texture Analysis

- Uses Local Binary Pattern (LBP) descriptors
- Analyzes skin texture variance
- Distinguishes real skin from printed photos/screens
- Threshold: texture_score > 0.5 indicates real face

### Motion Analysis

- Tracks facial landmark positions over time
- Detects micro-movements characteristic of live faces
- Uses optical flow analysis for motion verification
- Threshold: motion_score > 0.05 indicates natural movement

### Challenge-Response System

| Challenge | Description | Timeout |
|-----------|-------------|---------|
| BLINK | User must blink eyes | 50 frames |
| NOD | User must nod head | 50 frames |

### Liveness Score Calculation

```
Liveness Score = (Challenge × 40) + (Motion × 30) + (Texture × 30)
```

- **VALID**: Score ≥ 50
- **INVALID**: Score < 50 (potential spoofing detected)

---

## 📡 API Reference

### Web Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/run_script` | POST | Execute Python scripts |
| `/status` | GET | Get current process status |
| `/stop_script` | POST | Stop running process |
| `/attendance` | GET | View attendance records |
| `/api/attendance` | GET | JSON attendance data |

### Script Execution API

**POST /run_script**
```json
{
  "script": "get_faces | extract_features | attendance"
}
```

**Response:**
```json
{
  "status": "success | error",
  "message": "Started script_name | Error message"
}
```

### Status API

**GET /status**
```json
{
  "running": true,
  "output": "Process console output...",
  "script": "attendance_taker.py"
}
```

---

## 💾 Database Schema

### Attendance Table

```sql
CREATE TABLE attendance (
    name TEXT,           -- Person's name
    time TEXT,           -- Timestamp (HH:MM:SS)
    date DATE,           -- Date (YYYY-MM-DD)
    status TEXT,         -- VALID or INVALID
    UNIQUE(name, date)   -- One attendance per person per day
);
```

### Features CSV Structure

| Column | Type | Description |
|--------|------|-------------|
| Column 0 | String | Person name/ID |
| Columns 1-128 | Float | 128-dimensional face descriptor |

---

## ⚙ Configuration

### Anti-Spoofing Parameters

Located in `attendance_taker.py`:

```python
EYE_AR_THRESH = 0.25        # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 3    # Consecutive frames for blink
LBP_RADIUS = 3              # Local Binary Pattern radius
MOTION_FRAMES = 10          # Frames for motion analysis
MOTION_THRESHOLD = 0.05     # Motion detection threshold
CHALLENGE_DURATION = 50     # Frames to complete challenge
```

### Recognition Parameters

```python
FACE_CONFIDENCE_THRESHOLD = 0.4  # Recognition confidence
MAX_FACE_DISTANCE = 0.6          # Maximum Euclidean distance
```

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>Camera not detected</b></summary>

- Check if camera is connected and accessible
- Try different camera index: modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
- Ensure no other application is using the camera
</details>

<details>
<summary><b>dlib installation fails</b></summary>

- Install CMake: `brew install cmake` (macOS) or `apt install cmake` (Linux)
- Install build tools for your platform
- Try pre-built wheel: `pip install dlib-binary`
</details>

<details>
<summary><b>Face not recognized</b></summary>

- Ensure face was registered under good lighting
- Re-register with more photos from different angles
- Check if `features_all.csv` was generated properly
- Verify camera quality and positioning
</details>

<details>
<summary><b>Anti-spoofing too strict</b></summary>

- Adjust `LIVENESS_THRESHOLD` in configuration
- Ensure camera has adequate frame rate (30+ FPS)
- Improve lighting conditions
</details>

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests (if available)
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Areas for Contribution

- [ ] Add unit tests for core functions
- [ ] Implement additional anti-spoofing methods
- [ ] Add support for multiple cameras
- [ ] Create mobile-responsive improvements
- [ ] Add data visualization dashboards
- [ ] Implement user authentication for web interface

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[dlib](http://dlib.net/)** - Machine learning toolkit for face detection and recognition
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[Flask](https://flask.palletsprojects.com/)** - Python web framework
- **[Font Awesome](https://fontawesome.com/)** - Icon library
- Research papers on Eye Aspect Ratio and Local Binary Patterns for liveness detection

---

## 👨‍💻 Author

**SCI_CODER** - *Initial work and development*

- GitHub: [@amalssaienthusiast](https://github.com/amalssaienthusiast)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for secure attendance management

© 2025 TruePresence - Smart Face Recognition Attendance System

</div>
