# API Documentation

## TruePresence API Reference

This document provides detailed API documentation for the TruePresence Face Recognition Attendance System.

---

## Table of Contents

1. [Web API Endpoints](#web-api-endpoints)
2. [Python API Reference](#python-api-reference)
3. [Data Formats](#data-formats)
4. [Error Handling](#error-handling)
5. [Integration Examples](#integration-examples)

---

## Web API Endpoints

### Overview

Base URL: `http://localhost:5000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard page |
| `/run_script` | POST | Execute Python scripts |
| `/status` | GET | Get process status |
| `/stop_script` | POST | Stop running process |
| `/attendance` | GET | View attendance records |
| `/api/attendance` | GET | JSON attendance data |

---

### GET /

**Description:** Main dashboard page

**Response:** HTML page

---

### POST /run_script

**Description:** Execute one of the main Python scripts

**Request Body:**
```json
{
  "script": "get_faces | extract_features | attendance"
}
```

| Script Value | Executes |
|--------------|----------|
| `get_faces` | `get_faces_from_camera.py` (then auto-runs feature extraction) |
| `extract_features` | `features_extraction_to_csv.py` |
| `attendance` | `attendance_taker.py` |

**Success Response:**
```json
{
  "status": "success",
  "message": "Started get_faces"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Invalid script specified"
}
```

---

### GET /status

**Description:** Get the current status of any running process

**Response:**
```json
{
  "running": true,
  "output": "Process console output text...",
  "script": "attendance_taker.py"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `running` | boolean | Whether a process is currently running |
| `output` | string | Accumulated console output from the process |
| `script` | string | Name of the currently running script (null if none) |

---

### POST /stop_script

**Description:** Terminate the currently running process

**Success Response:**
```json
{
  "status": "success",
  "message": "Process terminated"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "No process running"
}
```

---

### GET /attendance

**Description:** View attendance records as HTML page

**Response:** HTML page with attendance table

**Query Parameters:** None (filtering done client-side)

---

### GET /api/attendance

**Description:** Get attendance records as JSON

**Success Response:**
```json
{
  "2025-01-07": [
    {
      "name": "John Doe",
      "time": "09:15:32",
      "date": "2025-01-07",
      "status": "VALID"
    },
    {
      "name": "Jane Smith",
      "time": "09:22:15",
      "date": "2025-01-07",
      "status": "VALID"
    }
  ],
  "2025-01-06": [
    {
      "name": "John Doe",
      "time": "08:55:10",
      "date": "2025-01-06",
      "status": "INVALID - Challenge not completed"
    }
  ]
}
```

**Error Response:**
```json
{
  "error": "Database file not found. Please take attendance first."
}
```

---

## Python API Reference

### Face_Recognizer Class

**Location:** `attendance_taker.py`

#### Constructor

```python
Face_Recognizer()
```

Initializes the face recognizer with all detection models and trackers.

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `face_features_known_list` | list | Loaded face feature vectors |
| `face_name_known_list` | list | Names corresponding to features |
| `liveness_scores` | dict | Per-face liveness scores |
| `current_challenges` | dict | Active challenges per face |

---

#### Methods

##### get_face_database()

```python
def get_face_database(self) -> None
```

Loads face features from `data/features_all.csv`.

**Returns:** None (populates `face_features_known_list` and `face_name_known_list`)

---

##### return_euclidean_distance()

```python
@staticmethod
def return_euclidean_distance(feature_1: np.ndarray, feature_2: np.ndarray) -> float
```

Calculates Euclidean distance between two 128D feature vectors.

**Parameters:**
- `feature_1`: First face descriptor
- `feature_2`: Second face descriptor

**Returns:** Distance value (lower = more similar)

---

##### detect_blink()

```python
def detect_blink(self, shape: np.ndarray) -> tuple[bool, float, np.ndarray, np.ndarray]
```

Detects eye blinks using Eye Aspect Ratio.

**Parameters:**
- `shape`: 68-point facial landmarks

**Returns:**
- `bool`: Whether blink was detected
- `float`: Current EAR value
- `np.ndarray`: Left eye landmarks
- `np.ndarray`: Right eye landmarks

---

##### analyze_face_texture()

```python
def analyze_face_texture(self, image: np.ndarray) -> tuple[float, bool]
```

Analyzes face texture for liveness detection.

**Parameters:**
- `image`: Face ROI image (BGR)

**Returns:**
- `float`: Texture score
- `bool`: Whether texture indicates real face

---

##### analyze_face_motion()

```python
def analyze_face_motion(self, face_id: int, shape: np.ndarray) -> tuple[float, bool]
```

Analyzes facial landmark motion over time.

**Parameters:**
- `face_id`: Unique identifier for the face
- `shape`: Current facial landmarks

**Returns:**
- `float`: Motion score
- `bool`: Whether motion is natural

---

##### detect_liveness()

```python
def detect_liveness(
    self, 
    face_id: int, 
    img_rd: np.ndarray, 
    d: dlib.rectangle, 
    shape: np.ndarray
) -> tuple[bool, str]
```

Performs comprehensive liveness detection.

**Parameters:**
- `face_id`: Unique face identifier
- `img_rd`: Full frame image
- `d`: dlib face rectangle
- `shape`: Facial landmarks

**Returns:**
- `bool`: Whether face is live (not spoofed)
- `str`: Status message

---

##### attendance()

```python
def attendance(self, name: str) -> None
```

Records attendance to the database.

**Parameters:**
- `name`: Person's name

**Database Operation:**
```sql
INSERT OR REPLACE INTO attendance (name, time, date, status) 
VALUES (?, ?, ?, ?)
```

---

##### process()

```python
def process(
    self, 
    stream: cv2.VideoCapture, 
    video_label: tk.Label, 
    update_metrics_callback: callable
) -> None
```

Main processing loop for face recognition.

**Parameters:**
- `stream`: OpenCV VideoCapture object
- `video_label`: Tkinter label for video display
- `update_metrics_callback`: Function to update UI metrics

---

### FaceRegisterApp Class

**Location:** `get_faces_from_camera.py`

#### Constructor

```python
FaceRegisterApp()
```

Initializes the face registration GUI application.

---

#### Methods

##### create_face_folder()

```python
def create_face_folder(self) -> None
```

Creates a new folder for a person's face images.

**Directory Created:** `data/data_faces_from_camera/person_{id}_{name}/`

---

##### save_current_face()

```python
def save_current_face(self) -> None
```

Saves the current face region to the person's folder.

**File Saved:** `img_face_{count}.jpg`

---

### Feature Extraction Functions

**Location:** `features_extraction_to_csv.py`

##### return_128d_features()

```python
def return_128d_features(path_img: str) -> np.ndarray | int
```

Extracts 128D face descriptor from an image.

**Parameters:**
- `path_img`: Path to the image file

**Returns:**
- `np.ndarray`: 128-dimensional face descriptor
- `int`: 0 if no face detected

---

##### return_features_mean_personX()

```python
def return_features_mean_personX(path_face_personX: str) -> np.ndarray
```

Computes mean of all face descriptors for one person.

**Parameters:**
- `path_face_personX`: Path to person's image folder

**Returns:** 128D mean feature vector

---

## Data Formats

### features_all.csv

| Column | Type | Description |
|--------|------|-------------|
| 0 | string | Person name |
| 1-128 | float | 128D face descriptor |

**Example:**
```csv
John_Doe,0.123456,-0.234567,...,0.345678
Jane_Smith,-0.456789,0.567890,...,-0.678901
```

### Attendance Database Schema

```sql
CREATE TABLE attendance (
    name TEXT,
    time TEXT,
    date DATE,
    status TEXT,
    UNIQUE(name, date)
);
```

### Status Values

| Status | Meaning |
|--------|---------|
| `VALID` | Liveness verification passed |
| `INVALID - Challenge not completed` | User didn't complete challenge |
| `INVALID - Motion not natural` | Suspected video/animation |
| `INVALID - Fake texture detected` | Suspected photo/printout |

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 500 | Internal server error |

### Python Exceptions

```python
# Camera initialization failure
cv2.VideoCapture(0).isOpened() == False

# Face detection failure (graceful - returns empty list)
faces = detector(img, 1)  # faces = []

# Database errors
sqlite3.Error: "database is locked"
```

---

## Integration Examples

### cURL Commands

```bash
# Start attendance system
curl -X POST http://localhost:5000/run_script \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "script=attendance"

# Check status
curl http://localhost:5000/status

# Get attendance data
curl http://localhost:5000/api/attendance

# Stop process
curl -X POST http://localhost:5000/stop_script
```

### Python Integration

```python
import requests

BASE_URL = "http://localhost:5000"

def start_attendance():
    response = requests.post(f"{BASE_URL}/run_script", data={"script": "attendance"})
    return response.json()

def get_status():
    response = requests.get(f"{BASE_URL}/status")
    return response.json()

def get_attendance():
    response = requests.get(f"{BASE_URL}/api/attendance")
    return response.json()

def stop_process():
    response = requests.post(f"{BASE_URL}/stop_script")
    return response.json()

# Example usage
result = start_attendance()
print(result)  # {"status": "success", "message": "Started attendance"}
```

### JavaScript Integration

```javascript
// Fetch attendance records
async function fetchAttendance() {
    const response = await fetch('/api/attendance');
    const data = await response.json();
    return data;
}

// Start a script
async function startScript(scriptName) {
    const formData = new FormData();
    formData.append('script', scriptName);
    
    const response = await fetch('/run_script', {
        method: 'POST',
        body: formData
    });
    return response.json();
}

// Poll status
async function pollStatus() {
    const response = await fetch('/status');
    const data = await response.json();
    
    if (data.running) {
        console.log('Process output:', data.output);
        setTimeout(pollStatus, 1000);
    }
}
```

---

## Rate Limits

*Not implemented in current version.*

For production deployment, consider adding rate limiting:

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@app.route('/api/attendance')
@limiter.limit("60/minute")
def get_attendance_api():
    ...
```

---

*Last updated: January 2025*
