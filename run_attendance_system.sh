#!/bin/bash

# Face Recognition Attendance System Runner
echo "Starting Face Recognition Attendance System..."

# Navigate to the flask_app directory
cd "$(dirname "$0")/flask_app"

# Check if pip/pip3 is available
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
FLASK_INSTALLED=$($PIP_CMD list | grep -i flask || echo "")
if [ -z "$FLASK_INSTALLED" ]; then
    echo "Installing Flask..."
    $PIP_CMD install flask
fi

# Check if the Python scripts exist in the parent directory
PARENT_DIR="$(dirname "$0")"
REQUIRED_SCRIPTS=("attendance_taker.py" "get_faces_from_camera_tkinter.py" "features_extraction_to_csv.py")
MISSING_SCRIPTS=()

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$PARENT_DIR/$script" ]; then
        MISSING_SCRIPTS+=("$script")
    fi
done

if [ ${#MISSING_SCRIPTS[@]} -gt 0 ]; then
    echo "WARNING: The following required scripts were not found:"
    for script in "${MISSING_SCRIPTS[@]}"; do
        echo "  - $script"
    done
    echo "Please ensure these files are in the parent directory: $PARENT_DIR"
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Determine the Python command to use
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python first."
    exit 1
fi

# Launch the Flask app
echo "Launching Face Recognition Attendance System..."
$PYTHON_CMD app.py

# Keep terminal open if error occurs
if [ $? -ne 0 ]; then
    echo "Error occurred while running the application."
    read -p "Press Enter to exit..." -n 1 -r
fi 