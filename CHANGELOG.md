# Changelog

All notable changes to the TruePresence project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-01-07

### 🎉 Initial Release

The first public release of TruePresence - Face Recognition Attendance System with Anti-Spoofing Technology.

### Added

#### Core Features
- **Face Registration System** (`get_faces_from_camera.py`)
  - Tkinter-based GUI for face capture
  - Multi-angle face image capture
  - Support for unlimited registered users
  - Real-time face detection visualization

- **Feature Extraction** (`features_extraction_to_csv.py`)
  - 128-dimensional face descriptor extraction using dlib ResNet
  - Mean feature calculation for improved accuracy
  - CSV-based feature storage

- **Attendance System** (`attendance_taker.py`)
  - Real-time face recognition
  - Euclidean distance matching algorithm
  - SQLite database integration
  - Tkinter GUI with live metrics

#### Anti-Spoofing Technology
- **Eye Blink Detection**
  - Eye Aspect Ratio (EAR) algorithm
  - Configurable threshold and consecutive frame count
  
- **Texture Analysis**
  - Local Binary Pattern (LBP) based detection
  - Real skin vs. printed photo differentiation
  
- **Motion Analysis**
  - Facial landmark motion tracking
  - Natural movement verification
  
- **Challenge-Response System**
  - Random challenge generation (BLINK, NOD)
  - Time-limited verification

#### Web Interface (`flask_app/`)
- Modern responsive dashboard
- Real-time process monitoring
- Console output streaming
- Attendance records viewer
- Search and filter functionality
- CSV export capability
- Database synchronization

#### Documentation
- Comprehensive README.md
- Technical documentation
- Contributing guidelines
- Security policy
- MIT License

### Technical Specifications
- Python 3.8+ compatibility
- dlib 19.24.0 face recognition models
- OpenCV 4.7.0 image processing
- Flask 2.2.3 web framework
- SQLite3 database

### Dependencies
```
dlib==19.24.0
opencv-python==4.7.0.72
numpy==1.24.2
pandas==1.5.3
flask==2.2.3
scipy==1.10.1
scikit-learn==1.2.2
imutils==0.5.4
```

---

## [Unreleased]

### Planned Features
- [ ] Multi-camera support
- [ ] Admin authentication for web interface
- [ ] Attendance reports and analytics
- [ ] Email/SMS notifications
- [ ] REST API improvements
- [ ] Docker containerization
- [ ] Mobile app integration
- [ ] Depth camera support for enhanced anti-spoofing

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-01-07 | Initial public release |

---

## How to Upgrade

### From Development to 1.0.0

```bash
git pull origin main
pip install -r requirements.txt
```

---

## Deprecation Notices

*None at this time.*

---

## Contributors

- **SCI_CODER** - Initial development and design

See also the list of [contributors](https://github.com/amalssaienthusiast/True_Presence/contributors) who participated in this project.
