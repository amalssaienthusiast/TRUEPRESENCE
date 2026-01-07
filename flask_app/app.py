from flask import Flask, render_template, request, jsonify, redirect, url_for
import subprocess
import threading
import os
import time
import signal
import sys
import shutil
import sqlite3
import datetime
import pandas as pd

app = Flask(__name__)

# Get the absolute path to the parent directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global process variables to keep track of running scripts
current_process = None
process_output = ""
process_name = None

def run_script(script_name, auto_next=False, next_script=None):
    """Run a Python script and capture its output"""
    global current_process, process_output, process_name
    
    # Kill any existing process
    if current_process and current_process.poll() is None:
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        except:
            pass
    
    process_output = ""
    process_name = script_name
    
    # Construct the absolute path to the script
    script_path = os.path.join(PARENT_DIR, script_name)
    
    # Check if the script exists
    if not os.path.exists(script_path):
        process_output = f"Error: Script not found at {script_path}"
        return False
    
    # Start the process
    try:
        # Set working directory to the parent directory
        current_process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PARENT_DIR,  # Set the current working directory to the parent
            preexec_fn=os.setsid  # Create a new process group
        )
        
        # Function to monitor the process
        def monitor_process():
            global process_output
            for line in current_process.stdout:
                process_output += line
                print(line, end='')  # Print to server console
            
            # If process completes and auto_next is True, start the next script
            if auto_next and next_script and current_process.poll() is not None:
                time.sleep(1)  # Brief pause before starting next process
                run_script(next_script)
        
        # Start monitoring thread
        thread = threading.Thread(target=monitor_process)
        thread.daemon = True
        thread.start()
        
        return True
    except Exception as e:
        process_output = f"Error starting {script_name}: {str(e)}"
        return False

# Ensure the database file is synchronized
def sync_database():
    """Copy the database file to Flask app directory if it exists in the parent directory"""
    parent_db_path = os.path.join(PARENT_DIR, "attendance.db")
    flask_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")
    
    # If database exists in parent directory, copy it to Flask directory
    if os.path.exists(parent_db_path):
        try:
            shutil.copy2(parent_db_path, flask_db_path)
            print(f"Database synchronized from {parent_db_path} to {flask_db_path}")
            return True
        except Exception as e:
            print(f"Failed to sync database: {str(e)}")
            return False
    else:
        print("No database found in parent directory.")
        return False

# Get attendance records from database
def get_attendance_records():
    """Retrieve attendance records from the database"""
    # First sync the database to ensure we have the latest data
    sync_database()
    
    # Get the path to the database file
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")
    
    if not os.path.exists(db_path):
        return {"error": "Database file not found. Please take attendance first."}
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        # Query for all attendance records
        query = "SELECT name, time, date, status FROM attendance ORDER BY date DESC, time DESC"
        df = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        # Convert to list of dictionaries for JSON serialization
        records = df.to_dict('records')
        
        # Group by date for better display
        records_by_date = {}
        for record in records:
            date = record['date']
            if date not in records_by_date:
                records_by_date[date] = []
            records_by_date[date].append(record)
        
        return records_by_date
    except Exception as e:
        return {"error": f"Failed to retrieve attendance records: {str(e)}"}

@app.route('/')
def index():
    # Sync database when someone accesses the main page
    sync_database()
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def execute_script():
    script = request.form.get('script')
    
    if script == 'get_faces':
        # Run get_faces_from_camera_tkinter.py with auto-launch of features extraction
        success = run_script('get_faces_from_camera.py', True, 'features_extraction_to_csv.py')
    elif script == 'extract_features':
        success = run_script('features_extraction_to_csv.py')
    elif script == 'attendance':
        success = run_script('attendance_taker.py')
    else:
        return jsonify({'status': 'error', 'message': 'Invalid script specified'})
    
    if success:
        return jsonify({'status': 'success', 'message': f'Started {script}'})
    else:
        return jsonify({'status': 'error', 'message': process_output})

@app.route('/status')
def get_status():
    global current_process, process_output, process_name
    
    if current_process is None:
        return jsonify({
            'running': False,
            'output': '',
            'script': None
        })
    
    is_running = current_process.poll() is None
    
    # If process just completed, sync the database
    if not is_running and current_process.poll() is not None:
        sync_database()
    
    return jsonify({
        'running': is_running,
        'output': process_output,
        'script': process_name
    })

@app.route('/stop_script', methods=['POST'])
def stop_script():
    global current_process
    
    if current_process and current_process.poll() is None:
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            return jsonify({'status': 'success', 'message': 'Process terminated'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error stopping process: {str(e)}'})
    else:
        return jsonify({'status': 'error', 'message': 'No process running'})

@app.route('/attendance')
def view_attendance():
    """Route to display attendance records"""
    records = get_attendance_records()
    return render_template('attendance.html', records=records)

@app.route('/api/attendance')
def get_attendance_api():
    """API endpoint for attendance data"""
    records = get_attendance_records()
    return jsonify(records)

if __name__ == '__main__':
    # Ensure we can run the app from either the main directory or the flask_app directory
    parent_script_paths = [
        os.path.join(PARENT_DIR, 'attendance_taker.py'),
        os.path.join(PARENT_DIR, 'get_faces_from_camera.py'),
        os.path.join(PARENT_DIR, 'features_extraction_to_csv.py')
    ]
    
    # Check if all required scripts exist
    missing_scripts = [path for path in parent_script_paths if not os.path.exists(path)]
    if missing_scripts:
        print("WARNING: The following required scripts were not found:")
        for script in missing_scripts:
            print(f"  - {script}")
        print("Please ensure these files are in the parent directory.")
    
    # Initial database sync
    sync_database()
    
    # Run Flask app
    app.run(debug=True, threaded=True) 