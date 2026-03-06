"""
get_faces_from_camera.py — Face Registration System (PyQt6)

Captures face images from a live camera feed for later feature extraction
and recognition.  Uses PyQt6 — fully replaces the original tkinter implementation with a
fully equivalent PyQt6 GUI.

Workflow
--------
1. Enter a numeric ID and a full name, then click "Create Person Folder".
2. Align face inside the green guide box and press **Space** (or the button)
   to capture a snapshot.
3. Repeat step 2 until enough samples are collected (≥10 recommended).
4. Run features_extraction_to_csv.py to build the recognition model.
"""

import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QTextEdit, QGroupBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QMessageBox, QSizePolicy, QScrollBar,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

# macOS: prevent OpenCV from requesting camera auth via background run loop.
# Camera is opened from the main thread in FaceRegisterApp.__init__().
import os as _os
_os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

# ---------------------------------------------------------------------------
# dlib setup
# ---------------------------------------------------------------------------
detector = dlib.get_frontal_face_detector()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class FaceRegisterApp(QMainWindow):
    """PyQt6 face-registration application."""

    # Path where per-person image folders are stored
    PATH_PHOTOS = "data/data_faces_from_camera/"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Face Registration System")
        self.setMinimumSize(1200, 700)

        # ── state ────────────────────────────────────────────────────────────
        self.existing_faces_cnt      = 0
        self.ss_cnt                  = 0          # images saved for current person
        self.current_frame_faces_cnt = 0

        self.face_folder_created_flag = False
        self.out_of_range_flag        = False

        # last detected face bounding box (for saving the ROI)
        self._face_left   = 0
        self._face_top    = 0
        self._face_width  = 0
        self._face_height = 0

        self.current_frame: np.ndarray | None = None  # latest RGB frame

        # FPS smoothing
        self._frame_start_time = time.time()
        self._fps              = 0.0

        self.current_face_dir: str | None = None

        # ── build UI ─────────────────────────────────────────────────────────
        self._build_ui()

        # ── camera ───────────────────────────────────────────────────────────
        self.cap = self._init_camera()

        # ── data directory ───────────────────────────────────────────────────
        os.makedirs(self.PATH_PHOTOS, exist_ok=True)
        self._refresh_face_count()

        # ── camera timer (≈30 fps) ───────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_frame)
        self._timer.start(33)

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── left: camera feed ────────────────────────────────────────────────
        cam_group = QGroupBox("Camera Feed")
        cam_layout = QVBoxLayout(cam_group)
        self.camera_label = QLabel("Camera initialising…")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background:#111; color:#888;")
        cam_layout.addWidget(self.camera_label)
        root.addWidget(cam_group, stretch=3)

        # ── right: controls ──────────────────────────────────────────────────
        right_widget = QWidget()
        right_widget.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(self._build_info_panel())
        right_layout.addWidget(self._build_registration_panel())
        right_layout.addWidget(self._build_actions_panel())
        right_layout.addWidget(self._build_log_panel(), stretch=1)

        root.addWidget(right_widget, stretch=1)

    def _build_info_panel(self) -> QGroupBox:
        group = QGroupBox("System Information")
        grid  = QGridLayout(group)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("FPS:"),               0, 0)
        self.fps_label   = QLabel("0.00")
        grid.addWidget(self.fps_label,               0, 1)

        grid.addWidget(QLabel("Registered Persons:"), 1, 0)
        self.db_count_label = QLabel("0")
        grid.addWidget(self.db_count_label,          1, 1)

        grid.addWidget(QLabel("Faces Detected:"),    2, 0)
        self.face_count_label = QLabel("0")
        grid.addWidget(self.face_count_label,        2, 1)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color:red; font-weight:bold;")
        grid.addWidget(self.warning_label,           3, 0, 1, 2)
        return group

    def _build_registration_panel(self) -> QGroupBox:
        group  = QGroupBox("Registration Steps")
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QLabel("<b>Step 1 — Enter Person Details</b>"), 0, 0, 1, 2)

        layout.addWidget(QLabel("ID (numbers only):"), 1, 0)
        self.id_entry = QLineEdit()
        self.id_entry.setPlaceholderText("e.g. 1")
        self.id_entry.textChanged.connect(self._enforce_numeric_id)
        layout.addWidget(self.id_entry, 1, 1)

        layout.addWidget(QLabel("Full Name:"), 2, 0)
        self.name_entry = QLineEdit()
        self.name_entry.setPlaceholderText("e.g. Alice Smith")
        layout.addWidget(self.name_entry, 2, 1)

        self.create_folder_btn = QPushButton("📁  Create Person Folder")
        self.create_folder_btn.clicked.connect(self._create_face_folder)
        self.create_folder_btn.setStyleSheet(
            "background:#4CAF50; color:white; padding:8px; font-weight:bold; border-radius:4px;")
        layout.addWidget(self.create_folder_btn, 3, 0, 1, 2)

        layout.addWidget(QLabel("<b>Step 2 — Capture Face Images</b>"), 4, 0, 1, 2)

        self.capture_btn = QPushButton("📷  Capture Face  [Space]")
        self.capture_btn.clicked.connect(self._save_current_face)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet(
            "background:#2196F3; color:white; padding:8px; font-weight:bold; border-radius:4px;")
        layout.addWidget(self.capture_btn, 5, 0, 1, 2)

        self.capture_count_label = QLabel("Captured: 0 images")
        layout.addWidget(self.capture_count_label, 6, 0, 1, 2)
        return group

    def _build_actions_panel(self) -> QGroupBox:
        group  = QGroupBox("Actions")
        layout = QVBoxLayout(group)

        clear_btn = QPushButton("🗑  Clear All Registered Data")
        clear_btn.clicked.connect(self._clear_data)
        clear_btn.setStyleSheet(
            "background:#f44336; color:white; padding:7px; border-radius:4px;")
        layout.addWidget(clear_btn)

        exit_btn = QPushButton("✖  Exit")
        exit_btn.clicked.connect(self.close)
        exit_btn.setStyleSheet("padding:7px; border-radius:4px;")
        layout.addWidget(exit_btn)
        return group

    def _build_log_panel(self) -> QGroupBox:
        group  = QGroupBox("Activity Log")
        layout = QVBoxLayout(group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setStyleSheet("background:#f9f9f9;")
        layout.addWidget(self.log_text)
        return group

    # =========================================================================
    # Camera helpers
    # =========================================================================

    def _init_camera(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(
                self, "Camera Error",
                "Could not open camera.\n"
                "• Check the camera is connected.\n"
                "• No other app is using it.\n"
                "• macOS/Windows privacy permissions are granted.",
            )
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._log("Camera opened (640×480)")
        return cap

    def _process_frame(self) -> None:
        """Read one frame, detect faces, annotate and display."""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame_rgb.copy()

        # FPS smoothing
        now = time.time()
        dt  = now - self._frame_start_time
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt if dt > 0 else 0)
        self._frame_start_time = now
        self.fps_label.setText(f"{self._fps:.2f}")

        # Face detection
        faces = detector(frame_rgb, 0)
        self.current_frame_faces_cnt = len(faces)
        self.face_count_label.setText(str(len(faces)))
        self.out_of_range_flag = False

        for face in faces:
            margin = 20
            too_close = (
                face.left()   < margin or
                face.right()  > 640 - margin or
                face.top()    < margin or
                face.bottom() > 480 - margin
            )
            if too_close:
                self.out_of_range_flag = True
                color = (220, 50, 50)
            else:
                color = (50, 220, 50)
                self._face_left   = face.left()
                self._face_top    = face.top()
                self._face_width  = face.width()
                self._face_height = face.height()

            size = max(face.width(), face.height())
            cx   = face.left() + face.width()  // 2
            cy   = face.top()  + face.height() // 2

            # Outer white guide square
            cv2.rectangle(frame_rgb,
                          (cx - size // 2 - 6, cy - size // 2 - 6),
                          (cx + size // 2 + 6, cy + size // 2 + 6),
                          (255, 255, 255), 1)
            # Inner coloured square
            cv2.rectangle(frame_rgb,
                          (cx - size // 2, cy - size // 2),
                          (cx + size // 2, cy + size // 2),
                          color, 2)
            # Crosshair
            cv2.line(frame_rgb, (cx - 18, cy), (cx + 18, cy), color, 1)
            cv2.line(frame_rgb, (cx, cy - 18), (cx, cy + 18), color, 1)

            label = "OK" if not too_close else "Move back / re-centre"
            cv2.putText(frame_rgb, label, (face.left(), face.bottom() + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # HUD
        cv2.putText(frame_rgb, f"FPS: {self._fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame_rgb, f"Faces: {len(faces)}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Display in QLabel
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.camera_label.width(),
                self.camera_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # =========================================================================
    # Registration actions
    # =========================================================================

    def _enforce_numeric_id(self, text: str) -> None:
        """Strip non-digit characters from the ID field."""
        cleaned = "".join(c for c in text if c.isdigit())
        if cleaned != text:
            self.id_entry.blockSignals(True)
            self.id_entry.setText(cleaned)
            self.id_entry.blockSignals(False)

    def _refresh_face_count(self) -> None:
        try:
            folders = [
                d for d in os.listdir(self.PATH_PHOTOS)
                if os.path.isdir(os.path.join(self.PATH_PHOTOS, d))
                and d.startswith("person_")
            ]
            self.existing_faces_cnt = len(folders)
            self.db_count_label.setText(str(self.existing_faces_cnt))
        except Exception as e:
            self._log(f"Count error: {e}", error=True)

    def _create_face_folder(self) -> None:
        pid  = self.id_entry.text().strip()
        name = self.name_entry.text().strip()
        if not pid:
            QMessageBox.warning(self, "Input Required", "Please enter a numeric ID.")
            return
        if not name:
            QMessageBox.warning(self, "Input Required", "Please enter the person's full name.")
            return

        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        folder_path = os.path.join(self.PATH_PHOTOS, f"person_{pid}_{clean_name}")

        try:
            os.makedirs(folder_path, exist_ok=True)
            self.current_face_dir = folder_path
            # Count existing captures (resume-friendly)
            self.ss_cnt = len([
                f for f in os.listdir(folder_path)
                if f.startswith("face_") and f.endswith(".jpg")
            ])
            self.face_folder_created_flag = True
            self.capture_btn.setEnabled(True)
            self.capture_count_label.setText(f"Captured: {self.ss_cnt} images")
            self._refresh_face_count()
            self._log(f"Folder ready: {folder_path}")
            QMessageBox.information(
                self, "Ready",
                f"Person folder created for {name} (ID {pid}).\n"
                f"Press Space or the Capture button to record face images.",
            )
        except Exception as e:
            self._log(f"Folder creation failed: {e}", error=True)
            QMessageBox.critical(self, "Error", str(e))

    def _save_current_face(self) -> None:
        if not self.face_folder_created_flag:
            QMessageBox.warning(self, "No Folder", "Create a person folder first.")
            return
        if self.current_frame is None:
            return
        if self.current_frame_faces_cnt == 0:
            QMessageBox.warning(self, "No Face", "No face detected in frame.")
            return
        if self.current_frame_faces_cnt > 1:
            QMessageBox.warning(self, "Multiple Faces",
                                "Only one face should be visible when capturing.")
            return
        if self.out_of_range_flag:
            QMessageBox.warning(self, "Position",
                                "Face is too close to the edge — please centre it.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"face_{timestamp}_{self.ss_cnt + 1}.jpg"
        save_path = os.path.join(self.current_face_dir, filename)

        try:
            pad = 20
            x1  = max(0, self._face_left  - pad)
            y1  = max(0, self._face_top   - pad)
            x2  = min(self.current_frame.shape[1], self._face_left  + self._face_width  + pad)
            y2  = min(self.current_frame.shape[0], self._face_top   + self._face_height + pad)
            roi = self.current_frame[y1:y2, x1:x2]
            cv2.imwrite(save_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
            self.ss_cnt += 1
            self.capture_count_label.setText(f"Captured: {self.ss_cnt} images")
            self._log(f"Saved: {filename}")
            self.warning_label.setText("✓  Image saved!")
            self.warning_label.setStyleSheet("color:green; font-weight:bold;")
            QTimer.singleShot(1500, lambda: (
                self.warning_label.setText(""),
                self.warning_label.setStyleSheet("color:red; font-weight:bold;"),
            ))
        except Exception as e:
            self._log(f"Save error: {e}", error=True)
            QMessageBox.critical(self, "Save Error", str(e))

    def _clear_data(self) -> None:
        reply = QMessageBox.question(
            self, "Confirm",
            "Delete ALL registered face data and the features CSV?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            if os.path.exists(self.PATH_PHOTOS):
                shutil.rmtree(self.PATH_PHOTOS)
            os.makedirs(self.PATH_PHOTOS, exist_ok=True)
            csv_path = "data/features_all.csv"
            if os.path.exists(csv_path):
                os.remove(csv_path)
            self.existing_faces_cnt = 0
            self.db_count_label.setText("0")
            self._log("All face data cleared.")
        except Exception as e:
            self._log(f"Clear error: {e}", error=True)

    # =========================================================================
    # Logging
    # =========================================================================

    def _log(self, message: str, error: bool = False) -> None:
        ts    = time.strftime("%H:%M:%S")
        color = "red" if error else "#222"
        self.log_text.append(
            f'<span style="color:{color}">[{ts}] {message}</span>'
        )
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        logger.info(message)

    # =========================================================================
    # Qt overrides
    # =========================================================================

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._save_current_face()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FaceRegisterApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
