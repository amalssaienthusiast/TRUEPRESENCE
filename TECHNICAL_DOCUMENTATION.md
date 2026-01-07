# Technical Documentation

## TruePresence - Face Recognition Attendance System

**Version:** 1.0.0  
**Last Updated:** January 2025

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Modules](#2-core-modules)
3. [Face Detection & Recognition Pipeline](#3-face-detection--recognition-pipeline)
4. [Anti-Spoofing Implementation](#4-anti-spoofing-implementation)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Database Design](#6-database-design)
7. [Flask Web Application](#7-flask-web-application)
8. [Performance Considerations](#8-performance-considerations)
9. [Security Considerations](#9-security-considerations)
10. [Extending the System](#10-extending-the-system)

---

## 1. System Overview

### 1.1 Purpose

TruePresence is designed to provide secure, reliable attendance tracking using facial recognition technology while mitigating common attack vectors such as photo, video, or mask-based spoofing attempts.

### 1.2 Design Principles

- **Security First**: Multiple anti-spoofing layers prevent fraudulent attendance
- **Modularity**: Components can be used independently or through the web interface
- **Simplicity**: Minimal dependencies and straightforward deployment
- **Real-Time Processing**: Optimized for live camera feed processing

### 1.3 High-Level Components

| Component | File | Purpose |
|-----------|------|---------|
| Face Registration | `get_faces_from_camera.py` | Capture and store face images |
| Feature Extraction | `features_extraction_to_csv.py` | Generate 128D face descriptors |
| Attendance System | `attendance_taker.py` | Recognition + Anti-spoofing |
| Web Interface | `flask_app/app.py` | Process management & UI |

---

## 2. Core Modules

### 2.1 Face Registration Module

**File:** `get_faces_from_camera.py`  
**Lines:** 536  
**Class:** `FaceRegisterApp`

#### 2.1.1 Key Methods

```python
class FaceRegisterApp:
    def __init__(self):
        # Initialize camera, counters, and GUI
        
    def create_face_folder(self):
        # Create directory for new person: data/data_faces_from_camera/person_X_name
        
    def save_current_face(self):
        # Capture and save cropped face image
        
    def process(self):
        # Main loop: detect faces, draw rectangles, update GUI
```

#### 2.1.2 Directory Structure Created

```
data/
└── data_faces_from_camera/
    ├── person_1_John/
    │   ├── img_face_1.jpg
    │   ├── img_face_2.jpg
    │   └── ...
    └── person_2_Jane/
        └── ...
```

#### 2.1.3 Face Detection

Uses dlib's frontal face detector:

```python
detector = dlib.get_frontal_face_detector()
faces = detector(img, 1)  # 1 = upsample factor
```

### 2.2 Feature Extraction Module

**File:** `features_extraction_to_csv.py`  
**Lines:** 94  
**Functions:** `return_128d_features()`, `return_features_mean_personX()`

#### 2.2.1 128-Dimensional Face Encoding

```python
# Load pre-trained models
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)
    shape = predictor(img_rd, faces[0])
    face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    return face_descriptor  # 128-dimensional vector
```

#### 2.2.2 Mean Feature Calculation

For each registered person, the system computes the mean of all captured face descriptors:

```python
def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    for photo in photos_list:
        features_128d = return_128d_features(path + photo)
        features_list_personX.append(features_128d)
    return np.array(features_list_personX).mean(axis=0)
```

### 2.3 Attendance Module

**File:** `attendance_taker.py`  
**Lines:** 737  
**Classes:** `Face_Recognizer`, `FaceRecognitionApp`

#### 2.3.1 Key Methods

```python
class Face_Recognizer:
    def get_face_database(self):
        # Load face features from CSV
        
    def return_euclidean_distance(feature_1, feature_2):
        # Calculate distance between face encodings
        
    def detect_liveness(self, face_id, img_rd, d, shape):
        # Multi-factor anti-spoofing check
        
    def attendance(self, name):
        # Record attendance to SQLite database
        
    def process(self, stream, video_label, update_metrics_callback):
        # Main recognition loop
```

---

## 3. Face Detection & Recognition Pipeline

### 3.1 Pipeline Stages

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Camera     │────▶│  dlib Face   │────▶│  68-Point       │
│  Capture    │     │  Detector    │     │  Landmark Det.  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
┌─────────────┐     ┌──────────────┐     ┌────────▼────────┐
│  Attendance │◀────│  Database    │◀────│  128D Feature   │
│  Record     │     │  Match       │     │  Extraction     │
└─────────────┘     └──────────────┘     └─────────────────┘
```

### 3.2 Face Matching Algorithm

The system uses Euclidean distance for face matching:

```python
@staticmethod
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist
```

**Matching Threshold:** Distance < 0.4 considered a match

### 3.3 Recognition Flow

```python
# For each detected face
for face in faces:
    # Extract 128D features
    shape = predictor(img, face)
    face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
    
    # Compare with database
    for stored_features in database:
        distance = return_euclidean_distance(face_descriptor, stored_features)
        if distance < threshold:
            # Match found
            record_attendance(name)
```

---

## 4. Anti-Spoofing Implementation

### 4.1 Eye Blink Detection

**Algorithm:** Eye Aspect Ratio (EAR)

```python
def eye_aspect_ratio(self, eye):
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])  # p2 to p6
    B = dist.euclidean(eye[2], eye[4])  # p3 to p5
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])  # p1 to p4
    
    ear = (A + B) / (2.0 * C)
    return ear
```

**Parameters:**
- `EYE_AR_THRESH = 0.25` - EAR below this = eye closed
- `EYE_AR_CONSEC_FRAMES = 3` - Consecutive frames to register blink

**Eye Landmarks (68-point model):**
- Left eye: points 36-41
- Right eye: points 42-47

### 4.2 Texture Analysis

**Algorithm:** Local Binary Pattern (LBP)

```python
def get_texture_features(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute LBP
    radius = LBP_RADIUS  # 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    # Calculate histogram as texture descriptor
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3))
    return hist / hist.sum()

def analyze_face_texture(self, image):
    descriptors = self.get_texture_features(image)
    texture_variance = np.var(descriptors)
    texture_score = texture_variance * 1000
    is_real = texture_score > 0.5  # Threshold for real skin texture
    return texture_score, is_real
```

**Principle:** Real skin has more texture variance than printed photos or screens.

### 4.3 Motion Analysis

**Algorithm:** Facial landmark motion tracking

```python
def analyze_face_motion(self, face_id, shape):
    current_landmarks = np.array([[shape.part(i).x, shape.part(i).y] 
                                   for i in range(68)])
    
    if face_id not in self.landmarks_history:
        self.landmarks_history[face_id] = deque(maxlen=MOTION_FRAMES)
    
    self.landmarks_history[face_id].append(current_landmarks)
    
    if len(self.landmarks_history[face_id]) >= 2:
        motion = np.mean(np.abs(
            self.landmarks_history[face_id][-1] - 
            self.landmarks_history[face_id][-2]
        ))
        is_natural = motion > MOTION_THRESHOLD
        return motion, is_natural
```

**Principle:** Live faces exhibit micro-movements; static images don't.

### 4.4 Challenge-Response System

```python
def generate_challenge(self, face_id):
    challenge = random.choice(CHALLENGE_TYPES)  # ["BLINK", "NOD"]
    self.current_challenges[face_id] = {
        'type': challenge,
        'start_frame': self.frame_cnt,
        'completed': False
    }
    return challenge

def check_challenge_response(self, face_id, shape):
    challenge = self.current_challenges[face_id]
    
    if challenge['type'] == 'BLINK':
        blink_detected, _, _, _ = self.detect_blink(shape)
        if blink_detected:
            challenge['completed'] = True
            
    elif challenge['type'] == 'NOD':
        nod_detected = self.detect_nod(face_id, shape)
        if nod_detected:
            challenge['completed'] = True
    
    return challenge['completed']
```

### 4.5 Liveness Score Calculation

```python
def detect_liveness(self, face_id, img_rd, d, shape):
    liveness_score = 0
    
    # Challenge response: 40 points
    if self.check_challenge_response(face_id, shape):
        liveness_score += 40
    
    # Motion analysis: 30 points
    motion_score, is_real_motion = self.analyze_face_motion(face_id, shape)
    if is_real_motion:
        liveness_score += min(30, motion_score * 200)
    
    # Texture analysis: 30 points
    texture_score, is_real_texture = self.analyze_face_texture(face_roi)
    if is_real_texture:
        liveness_score += min(30, texture_score)
    
    # Exponential moving average smoothing
    alpha = 0.3
    self.liveness_scores[face_id] = alpha * liveness_score + (1 - alpha) * self.liveness_scores.get(face_id, 0)
    
    is_live = self.liveness_scores[face_id] >= 50
    return is_live, spoof_message
```

---

## 5. Data Flow Architecture

### 5.1 Registration Flow

```
User Input ──▶ Camera Capture ──▶ Face Detection ──▶ ROI Cropping
                                                          │
                                                          ▼
data/data_faces_from_camera/person_X_name/ ◀── Save Image
```

### 5.2 Feature Extraction Flow

```
Face Images ──▶ dlib Detector ──▶ 68 Landmarks ──▶ ResNet Model
      │                                                 │
      │                                                 ▼
      └──────────────────────────────────▶ data/features_all.csv
                                          (name, 128D vector)
```

### 5.3 Attendance Flow

```
Camera Feed ──▶ Detection ──▶ Recognition ──▶ Anti-Spoofing
                                                    │
                          ┌──────────▲──────────────┘
                          │          │
                 ┌────────┴──────────┴────────┐
                 │                            │
             VALID                        INVALID
                 │                            │
                 ▼                            ▼
         attendance.db                 attendance.db
         (status=VALID)               (status=INVALID)
```

---

## 6. Database Design

### 6.1 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS attendance (
    name TEXT,
    time TEXT,
    date DATE,
    status TEXT,
    UNIQUE(name, date)  -- One attendance per person per day
);
```

### 6.2 Data Integrity

- **UNIQUE constraint** prevents duplicate attendance entries
- **REPLACE INTO** used for upsert operations
- Database synchronized between main and Flask directories

### 6.3 Query Examples

```sql
-- Get all attendance for a specific date
SELECT * FROM attendance WHERE date = '2025-01-07' ORDER BY time DESC;

-- Get attendance summary by person
SELECT name, COUNT(*) as days_present FROM attendance 
WHERE status = 'VALID' GROUP BY name;

-- Get invalid attempts (potential spoofing)
SELECT * FROM attendance WHERE status LIKE '%INVALID%';
```

---

## 7. Flask Web Application

### 7.1 Application Structure

```
flask_app/
├── app.py              # Main Flask application
├── static/
│   ├── css/
│   │   └── styles.css  # Styling
│   └── js/
│       └── main.js     # Client-side logic
├── templates/
│   ├── index.html      # Dashboard
│   └── attendance.html # Records view
└── requirements.txt
```

### 7.2 Process Management

```python
def run_script(script_name, auto_next=False, next_script=None):
    global current_process, process_output, process_name
    
    # Kill any existing process
    if current_process and current_process.poll() is None:
        os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
    
    # Start subprocess with output capture
    current_process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PARENT_DIR,
        preexec_fn=os.setsid  # New process group
    )
    
    # Monitor in background thread
    thread = threading.Thread(target=monitor_process)
    thread.daemon = True
    thread.start()
```

### 7.3 Database Synchronization

```python
def sync_database():
    """Copy database from parent directory to Flask directory"""
    parent_db = os.path.join(PARENT_DIR, "attendance.db")
    flask_db = os.path.join(os.path.dirname(__file__), "attendance.db")
    
    if os.path.exists(parent_db):
        shutil.copy2(parent_db, flask_db)
        return True
    return False
```

### 7.4 API Endpoints

| Endpoint | Method | Handler | Response |
|----------|--------|---------|----------|
| `/` | GET | `index()` | HTML |
| `/run_script` | POST | `execute_script()` | JSON |
| `/status` | GET | `get_status()` | JSON |
| `/stop_script` | POST | `stop_script()` | JSON |
| `/attendance` | GET | `view_attendance()` | HTML |
| `/api/attendance` | GET | `get_attendance_api()` | JSON |

---

## 8. Performance Considerations

### 8.1 Frame Rate Optimization

- Target: 30 FPS for real-time processing
- Face detection runs every frame
- Texture analysis runs every 5th frame (costly operation)
- Feature comparison parallelized where possible

### 8.2 Memory Management

- `deque(maxlen=N)` for motion history (bounded memory)
- Face features loaded once and cached
- Camera frames processed in-place when possible

### 8.3 Bottlenecks

| Operation | Time Complexity | Optimization |
|-----------|-----------------|--------------|
| Face Detection | O(n²) | dlib HOG faster than CNN |
| 128D Extraction | O(face_area) | ResNet inference |
| Database Match | O(k) | k = registered faces |
| Anti-Spoofing | O(1) per-face | Cached scores |

---

## 9. Security Considerations

### 9.1 Attack Vectors Mitigated

| Attack | Mitigation |
|--------|------------|
| Photo presentation | Texture analysis, blink detection |
| Video playback | Motion analysis, challenge-response |
| 3D masks | Texture analysis, motion patterns |
| Replay attacks | Random challenge selection |

### 9.2 Limitations

- Not resistant to very high-quality 3D masks
- Lighting conditions affect performance
- Single-camera system (no depth sensing)

### 9.3 Recommendations

- Deploy with proper lighting
- Consider adding depth sensor for higher security
- Regularly update face registrations
- Monitor for INVALID status spikes

---

## 10. Extending the System

### 10.1 Adding New Anti-Spoofing Methods

1. Create new detection function in `Face_Recognizer`:
```python
def detect_new_method(self, face_id, image):
    # Implement detection logic
    return score, is_real
```

2. Integrate into `detect_liveness()`:
```python
if is_real:
    liveness_score += new_method_weight
```

### 10.2 Adding API Authentication

```python
from functools import wraps
from flask import request, abort

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            abort(401)
        return f(*args, **kwargs)
    return decorated

@app.route('/api/attendance')
@require_api_key
def get_attendance_api():
    ...
```

### 10.3 Multi-Camera Support

```python
class MultiFaceRecognizer:
    def __init__(self, camera_ids=[0, 1]):
        self.recognizers = [Face_Recognizer(cam_id) for cam_id in camera_ids]
    
    def run_all(self):
        threads = [threading.Thread(target=r.run) for r in self.recognizers]
        for t in threads:
            t.start()
```

---

## Appendix A: File Reference

| File | Lines | Description |
|------|-------|-------------|
| `attendance_taker.py` | 737 | Main recognition + anti-spoofing |
| `get_faces_from_camera.py` | 536 | Face registration GUI |
| `features_extraction_to_csv.py` | 94 | Feature extraction |
| `flask_app/app.py` | 229 | Web application |
| `flask_app/templates/index.html` | 89 | Dashboard template |
| `flask_app/templates/attendance.html` | 207 | Records template |

## Appendix B: Configuration Reference

```python
# attendance_taker.py
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
LBP_RADIUS = 3
MOTION_FRAMES = 10
MOTION_THRESHOLD = 0.05
CHALLENGE_ACTIVE = True
CHALLENGE_TYPES = ["BLINK", "NOD"]
CHALLENGE_DURATION = 50
```

---

*Document generated: January 2025*  
*TruePresence v1.0.0*
