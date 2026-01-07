# Face Recognition Attendance System - Flask Web Interface

This Flask web application provides a modern, user-friendly interface for the Face Recognition Attendance System with anti-spoofing technology.

## Features

- Responsive web interface for all devices
- Integration with existing face recognition scripts
- Real-time process monitoring
- Automatic workflow between face registration and feature extraction
- Clean visualization of attendance results
- Database synchronization between main and Flask app directories

## Setup Instructions

1. **File Structure**:
   Make sure you have the following structure:
   ```
   project-root/
   ├── attendance_taker.py
   ├── get_faces_from_camera_tkinter.py
   ├── features_extraction_to_csv.py
   ├── flask_app/
   │   ├── app.py
   │   ├── static/
   │   ├── templates/
   │   └── ...
   ```

2. **Install Python dependencies**:
   ```
   pip install flask
   ```

3. **Start the Flask application**:
   You can run the app from either location:
   
   From project root:
   ```
   cd flask_app
   python app.py
   ```
   
   Or directly from flask_app directory:
   ```
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. **Register New Face**: Captures and saves face images for recognition.
   - After completion, feature extraction will start automatically.

2. **Extract Features**: Processes captured face images to extract identification features.
   - This step is usually automatic after registration.

3. **Take Attendance**: Starts the attendance system with anti-spoofing verification.
   - Records attendance in the SQLite database.

## Process Monitoring

The console output section shows real-time logs from the running processes. Use the "Stop Process" button to terminate the current operation if needed.

## Database Synchronization

The system automatically synchronizes the attendance database file between the main directory and the Flask app directory. This ensures that:
- The web interface can access the latest attendance data
- The original Python scripts operate on the same database when run directly

## System Requirements

- Python 3.6 or higher
- Flask
- All dependencies required by the original face recognition scripts (dlib, OpenCV, etc.)
- Modern web browser with JavaScript enabled

## Troubleshooting

If you encounter issues:

1. **Script Not Found Errors**:
   - Make sure all three Python scripts (attendance_taker.py, get_faces_from_camera_tkinter.py, features_extraction_to_csv.py) are in the main directory

2. **Database Issues**:
   - Check that both the main directory and flask_app directory have write permissions
   - If using the app for the first time, either script will create the database file

3. **Other Common Issues**:
   - Make sure all Python dependencies are installed
   - Check browser console for any JavaScript errors
   - Look at the Flask server output for Python errors 