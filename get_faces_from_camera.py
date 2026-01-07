import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

class FaceRegisterApp:
    def __init__(self):
        # Initialize counters
        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        
        # Setup main window with light theme
        self.win = tk.Tk()
        self.win.title("Face Registration System")
        self.win.geometry("1200x700")
        self.win.configure(bg='white')
        
        # Set theme and styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Setup window close behavior
        self.win.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # Camera and processing variables
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.out_of_range_flag = False
        self.face_folder_created_flag = False
        
        # FPS tracking
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0.0
        self.start_time = time.time()
        
        # Path configuration
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = None
        
        # Initialize GUI components
        self.setup_gui()
        
        # Initialize camera
        self.cap = self.initialize_camera()
        
        # Create necessary directories
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()

    def configure_styles(self):
        """Configure ttk styles for light theme"""
        self.style.configure('.', background='white', foreground='black')
        self.style.configure('TFrame', background='white')
        self.style.configure('TLabel', background='white', foreground='black')
        self.style.configure('TButton', background='#f0f0f0', foreground='black', 
                           borderwidth=1, focusthickness=3, focuscolor='none')
        self.style.configure('TEntry', fieldbackground='white')
        self.style.configure('TLabelframe', background='white', foreground='#333333')
        self.style.configure('TLabelframe.Label', background='white', foreground='#333333')
        self.style.map('TButton',
                     background=[('active', '#e0e0e0'), ('pressed', '#d0d0d0')],
                     foreground=[('active', 'black'), ('pressed', 'black')])

    def setup_gui(self):
        """Setup all GUI components"""
        # Main frames
        self.main_frame = ttk.Frame(self.win)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera frame (left)
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Camera Feed")
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(padx=10, pady=10)
        
        # Control frame (right)
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Information panel
        self.info_panel = ttk.LabelFrame(self.control_frame, text="System Information")
        self.info_panel.pack(fill=tk.X, pady=5)
        
        self.setup_info_panel()
        
        # Registration panel
        self.reg_panel = ttk.LabelFrame(self.control_frame, text="Registration Steps")
        self.reg_panel.pack(fill=tk.X, pady=5)
        
        self.setup_registration_panel()
        
        # Actions panel
        self.actions_panel = ttk.LabelFrame(self.control_frame, text="Actions")
        self.actions_panel.pack(fill=tk.X, pady=5)
        
        self.setup_actions_panel()
        
        # Log panel
        self.log_panel = ttk.LabelFrame(self.control_frame, text="Activity Log")
        self.log_panel.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.setup_log_panel()

    def setup_info_panel(self):
        """Setup the information display panel"""
        # FPS display
        ttk.Label(self.info_panel, text="FPS:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.fps_label = ttk.Label(self.info_panel, text="0.00")
        self.fps_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Database count
        ttk.Label(self.info_panel, text="Registered Persons:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.db_count_label = ttk.Label(self.info_panel, text="0")
        self.db_count_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Faces in frame
        ttk.Label(self.info_panel, text="Faces Detected:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.face_count_label = ttk.Label(self.info_panel, text="0")
        self.face_count_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Warning label
        self.warning_label = ttk.Label(self.info_panel, text="", foreground="red")
        self.warning_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

    def setup_registration_panel(self):
        """Setup the registration controls"""
        # Step 1: Input details
        ttk.Label(self.reg_panel, text="Step 1: Enter Person Details").grid(
            row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(5, 2))
        
        ttk.Label(self.reg_panel, text="ID (numbers only):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.id_entry = ttk.Entry(self.reg_panel, validate="key")
        self.id_entry['validatecommand'] = (self.id_entry.register(self.validate_id), '%P')
        self.id_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        ttk.Label(self.reg_panel, text="Full Name:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.name_entry = ttk.Entry(self.reg_panel)
        self.name_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        
        self.create_folder_btn = ttk.Button(
            self.reg_panel, text="Create Person Folder", 
            command=self.create_face_folder, style='Accent.TButton')
        self.create_folder_btn.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # Step 2: Capture faces
        ttk.Label(self.reg_panel, text="Step 2: Capture Face Images").grid(
            row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 2))
        
        self.capture_btn = ttk.Button(
            self.reg_panel, text="Capture Face (Space)", 
            command=self.save_current_face, state='disabled')
        self.capture_btn.grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # Bind space key to capture
        self.win.bind('<space>', lambda event: self.save_current_face())

    def setup_actions_panel(self):
        """Setup action buttons"""
        self.clear_data_btn = ttk.Button(
            self.actions_panel, text="Clear All Data", 
            command=self.clear_data)
        self.clear_data_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.exit_btn = ttk.Button(
            self.actions_panel, text="Exit", 
            command=self.close_app)
        self.exit_btn.pack(fill=tk.X, padx=5, pady=5)

    def setup_log_panel(self):
        """Setup the logging text area"""
        self.log_text = tk.Text(
            self.log_panel, height=10, state='disabled', 
            bg='white', fg='black', wrap=tk.WORD)
        
        scrollbar = ttk.Scrollbar(
            self.log_panel, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = scrollbar.set
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def validate_id(self, text):
        """Validate that ID contains only numbers"""
        if text == "":
            return True
        try:
            int(text)
            return True
        except ValueError:
            return False

    def initialize_camera(self):
        """Initialize the camera with error handling"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_message("ERROR: Could not open camera.", error=True)
            messagebox.showerror(
                "Camera Error", 
                "Could not open camera. Please check:\n"
                "1. Camera is connected\n"
                "2. No other application is using the camera\n"
                "3. Permissions are granted")
            self.win.after(100, self.win.destroy)
            return None
        
        # Set camera resolution to 640x480 for consistency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def log_message(self, message, error=False):
        """Add a message to the log with timestamp"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state='normal')
        if error:
            self.log_text.tag_config('error', foreground='red')
            self.log_text.insert(tk.END, log_entry, 'error')
        else:
            self.log_text.insert(tk.END, log_entry)
            
        self.log_text.config(state='disabled')
        self.log_text.see(tk.END)
        logging.info(message)

    def pre_work_mkdir(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.path_photos_from_camera, exist_ok=True)
        self.log_message("Initialized data directory")

    def check_existing_faces_cnt(self):
        """Count existing registered persons"""
        try:
            if os.path.isdir(self.path_photos_from_camera):
                person_folders = [d for d in os.listdir(self.path_photos_from_camera)
                                 if os.path.isdir(os.path.join(self.path_photos_from_camera, d)) 
                                 and d.startswith("person_")]
                self.existing_faces_cnt = len(person_folders)
            else:
                self.existing_faces_cnt = 0
                
            self.db_count_label['text'] = str(self.existing_faces_cnt)
        except Exception as e:
            self.log_message(f"Error counting existing faces: {str(e)}", error=True)
            self.existing_faces_cnt = 0

    def update_fps(self):
        """Calculate and update FPS display"""
        now = time.time()
        if now > self.start_time:
            self.frame_time = now - self.frame_start_time
            if self.frame_time > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / self.frame_time)  # Smoothing
        
        self.frame_start_time = now
        
        # Update display every second
        if int(self.start_time) != int(now):
            self.fps_label["text"] = f"{self.fps:.2f}"
        self.start_time = now

    def create_face_folder(self):
        """Create a folder for a new person"""
        person_id = self.id_entry.get().strip()
        person_name = self.name_entry.get().strip()
        
        # Validation
        if not person_id:
            messagebox.showwarning("Input Required", "Please enter a numeric ID.")
            return
        if not person_name:
            messagebox.showwarning("Input Required", "Please enter a name.")
            return
        
        # Create folder name
        clean_name = "".join(c if c.isalnum() else "_" for c in person_name)
        self.current_face_dir = os.path.join(
            self.path_photos_from_camera, 
            f"person_{person_id}_{clean_name}")
        
        try:
            os.makedirs(self.current_face_dir, exist_ok=True)
            self.log_message(f"Created folder: {self.current_face_dir}")
            
            # Count existing images in folder
            existing_files = [f for f in os.listdir(self.current_face_dir) 
                            if f.startswith("face_") and f.endswith(".jpg")]
            self.ss_cnt = len(existing_files)
            
            self.face_folder_created_flag = True
            self.capture_btn['state'] = 'normal'
            self.check_existing_faces_cnt()
            
            messagebox.showinfo(
                "Success", 
                f"Folder created for {person_name} (ID: {person_id}).\n"
                f"You can now capture face images.")
                
        except Exception as e:
            self.log_message(f"Error creating folder: {str(e)}", error=True)
            messagebox.showerror(
                "Error", 
                f"Could not create folder:\n{str(e)}")

    def save_current_face(self):
        """Save the current face image"""
        if not self.face_folder_created_flag:
            messagebox.showwarning(
                "No Folder", 
                "Please create a person folder first.")
            return
            
        if self.current_frame_faces_cnt != 1:
            if self.current_frame_faces_cnt == 0:
                message = "No face detected in frame."
            else:
                message = "Multiple faces detected. Please ensure only one face is visible."
            messagebox.showwarning("Cannot Save", message)
            return
            
        if self.out_of_range_flag:
            messagebox.showwarning(
                "Position Error", 
                "Face is too close to edge of frame.\n"
                "Please center the face in the frame.")
            return
            
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"face_{timestamp}_{self.ss_cnt + 1}.jpg"
        save_path = os.path.join(self.current_face_dir, filename)
        
        try:
            # Get face region with some padding
            padding = 20  # pixels around face
            x1 = max(0, self.face_ROI_width_start - padding)
            y1 = max(0, self.face_ROI_height_start - padding)
            x2 = min(self.current_frame.shape[1], self.face_ROI_width_start + self.face_ROI_width + padding)
            y2 = min(self.current_frame.shape[0], self.face_ROI_height_start + self.face_ROI_height + padding)
            
            # Extract and save face
            face_img = self.current_frame[y1:y2, x1:x2]
            cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            self.ss_cnt += 1
            self.log_message(f"Saved face image: {filename}")
            
            # Visual feedback
            self.warning_label['text'] = "FACE SAVED!"
            self.warning_label['foreground'] = 'green'
            self.win.after(1000, lambda: self.warning_label.config(text=""))
            
        except Exception as e:
            self.log_message(f"Error saving face: {str(e)}", error=True)
            messagebox.showerror("Save Error", f"Could not save image:\n{str(e)}")

    def get_frame(self):
        """Capture frame from camera with error handling"""
        if not self.cap:
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ret, None
        except Exception as e:
            self.log_message(f"Camera error: {str(e)}", error=True)
            return False, None

    def process(self):
        """Main processing loop for face detection"""
        if not self.cap:
            return
            
        ret, self.current_frame = self.get_frame()
        
        if ret and self.current_frame is not None:
            self.update_fps()
            
            # Detect faces using dlib
            faces = detector(self.current_frame, 0)
            self.current_frame_faces_cnt = len(faces)
            self.face_count_label['text'] = str(self.current_frame_faces_cnt)
            
            self.out_of_range_flag = False
            
            # Draw face rectangles
            if faces:
                for face in faces:
                    # Get face coordinates
                    self.face_ROI_width_start = face.left()
                    self.face_ROI_height_start = face.top()
                    self.face_ROI_width = face.width()
                    self.face_ROI_height = face.height()
                    
                    # Check if face is too close to edge
                    margin = 20
                    if (face.left() < margin or face.right() > (640 - margin) or
                        face.top() < margin or face.bottom() > (480 - margin)):
                        self.out_of_range_flag = True
                        color = (255, 0, 0)  # Red for out of range
                        thickness = 2
                    else:
                        color = (255, 255, 255)  # White for in range
                        thickness = 2
                    
                    # Draw square around face (with equal width/height)
                    size = max(face.width(), face.height())
                    center_x = face.left() + face.width() // 2
                    center_y = face.top() + face.height() // 2
                    
                    # Draw outer white square
                    cv2.rectangle(
                        self.current_frame,
                        (center_x - size//2 - 5, center_y - size//2 - 5),
                        (center_x + size//2 + 5, center_y + size//2 + 5),
                        (255, 255, 255), thickness)
                    
                    # Draw inner red square
                    cv2.rectangle(
                        self.current_frame,
                        (center_x - size//2, center_y - size//2),
                        (center_x + size//2, center_y + size//2),
                        color, thickness)
                    
                    # Add crosshair for precise alignment
                    cv2.line(
                        self.current_frame,
                        (center_x - 15, center_y),
                        (center_x + 15, center_y),
                        color, 1)
                    cv2.line(
                        self.current_frame,
                        (center_x, center_y - 15),
                        (center_x, center_y + 15),
                        color, 1)
            
            # Update warning label
            if self.out_of_range_flag:
                self.warning_label['text'] = "MOVE FACE TO CENTER"
                self.warning_label['foreground'] = 'red'
            else:
                self.warning_label['text'] = ""
            
            # Update camera display
            img = Image.fromarray(self.current_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        
        # Schedule next frame processing
        self.win.after(20, self.process)

    def clear_data(self):
        """Clear all registered face data"""
        if not messagebox.askyesno(
            "Confirm Clear", 
            "This will permanently delete ALL registered face data.\n"
            "Are you sure you want to continue?"):
            return
            
        try:
            # Delete all person folders
            for item in os.listdir(self.path_photos_from_camera):
                path = os.path.join(self.path_photos_from_camera, item)
                if os.path.isdir(path) and item.startswith("person_"):
                    shutil.rmtree(path)
            
            # Reset state
            self.current_face_dir = None
            self.face_folder_created_flag = False
            self.capture_btn['state'] = 'disabled'
            self.id_entry.delete(0, tk.END)
            self.name_entry.delete(0, tk.END)
            self.ss_cnt = 0
            
            self.check_existing_faces_cnt()
            self.log_message("Cleared all registered face data.")
            messagebox.showinfo("Success", "All registered face data has been deleted.")
            
        except Exception as e:
            self.log_message(f"Error clearing data: {str(e)}", error=True)
            messagebox.showerror("Error", f"Could not clear data:\n{str(e)}")

    def close_app(self):
        """Cleanup and close application"""
        self.log_message("Closing application...")
        if self.cap:
            self.cap.release()
        self.win.destroy()

    def run(self):
        """Start the application"""
        if self.cap:
            self.process()
        self.win.mainloop()

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='face_register.log',
        filemode='a')
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    
    # Create and run application
    app = FaceRegisterApp()
    app.run()

if __name__ == '__main__':
    main()