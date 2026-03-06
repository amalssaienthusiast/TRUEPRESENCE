"""
attendance_taker.py — Face Recognition + Anti-Spoofing Attendance System (PyQt6)

Anti-spoofing layers
--------------------
1. Eye-Blink Detection    — Eye Aspect Ratio (EAR) < threshold triggers blink
2. Motion Analysis        — landmark variance over N frames detects live motion
3. Texture Analysis       — HOG descriptor variance rejects static photos/screens
4. Challenge-Response     — user must BLINK or NOD within a time window
5. Composite Liveness     — smoothed score ≥ 50 required to mark VALID attendance

Attendance is recorded to PostgreSQL (via database.py / docker-compose).
"""

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import datetime
import random
import sys
from scipy.spatial import distance as dist
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
from database import init_db, record_attendance

# ---------------------------------------------------------------------------
# macOS AVFoundation: skip OpenCV's built-in auth request — the app opens
# the camera on the main thread, which is the correct place to trigger the
# macOS permission dialog.
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

# ---------------------------------------------------------------------------
# dlib models
# ---------------------------------------------------------------------------
detector      = dlib.get_frontal_face_detector()
predictor     = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1(
    "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
)

# ---------------------------------------------------------------------------
# Eye landmark indices
# ---------------------------------------------------------------------------
LEFT_EYE_START,  LEFT_EYE_END  = 42, 48
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42

# ---------------------------------------------------------------------------
# Anti-spoofing / liveness constants
# ---------------------------------------------------------------------------
EYE_AR_THRESH        = 0.30   # EAR below this → blink
EYE_AR_CONSEC_FRAMES = 1      # consecutive frames required
BLINK_REQUIRED       = False  # set True to enforce blink before VALID

MOTION_FRAMES     = 10
MOTION_THRESHOLD  = 0.05
CHALLENGE_ACTIVE  = True
CHALLENGE_TYPES   = ["BLINK", "NOD"]
CHALLENGE_DURATION = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Core face-recognition & anti-spoofing engine
# ===========================================================================

class Face_Recognizer:
    """Handles face detection, recognition, liveness analysis and attendance."""

    def __init__(self) -> None:
        self.font              = cv2.FONT_HERSHEY_SIMPLEX
        self.frame_cnt         = 0
        self.fps               = 0.0
        self._frame_start      = time.time()

        # Known faces loaded from CSV
        self.face_features_known_list: list = []
        self.face_name_known_list:     list = []

        # Per-frame tracking
        self.last_frame_face_centroid_list:  list = []
        self.current_frame_face_centroid_list: list = []
        self.last_frame_face_name_list:      list = []
        self.current_frame_face_name_list:   list = []
        self.last_frame_face_cnt             = 0
        self.current_frame_face_cnt          = 0
        self.current_frame_face_X_e_distance_list: list = []
        self.current_frame_face_position_list:     list = []
        self.current_frame_face_feature_list:      list = []
        self.last_current_frame_centroid_e_distance = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval     = 10

        # Blink state
        self.eye_counter    = 0
        self.total_blinks   = 0
        self.blink_detected = False

        # Per-person state
        self.person_blink_status: dict[str, bool]  = {}
        self.face_motion_history:  dict             = {}
        self.liveness_scores:      dict[str, float] = {}
        self.spoof_detected:       dict[str, bool]  = {}
        self.current_challenges:   dict[str, str]   = {}
        self.challenge_progress:   dict[str, int]   = {}
        self.challenge_complete:   dict[str, bool]  = {}
        self.prev_landmarks:       dict             = {}

    # ── eye aspect ratio ───────────────────────────────────────────────────

    def eye_aspect_ratio(self, eye: list) -> float:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def get_eyes(self, shape):
        left  = [(shape.part(i).x, shape.part(i).y) for i in range(LEFT_EYE_START,  LEFT_EYE_END)]
        right = [(shape.part(i).x, shape.part(i).y) for i in range(RIGHT_EYE_START, RIGHT_EYE_END)]
        return left, right

    def detect_blink(self, shape):
        left, right = self.get_eyes(shape)
        ear = (self.eye_aspect_ratio(left) + self.eye_aspect_ratio(right)) / 2.0
        if ear < EYE_AR_THRESH:
            self.eye_counter += 1
            if self.eye_counter >= EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
                self.blink_detected = True
        else:
            self.eye_counter = 0
        return self.blink_detected, ear, left, right

    # ── feature database ───────────────────────────────────────────────────

    def get_face_database(self) -> bool:
        csv_path = "data/features_all.csv"
        if not os.path.exists(csv_path):
            logger.warning("'features_all.csv' not found — run feature extraction first.")
            return False
        try:
            df = pd.read_csv(csv_path, header=None)
            for i in range(df.shape[0]):
                self.face_name_known_list.append(df.iloc[i][0])
                feats = [str(df.iloc[i][j]) if df.iloc[i][j] != "" else "0" for j in range(1, 129)]
                self.face_features_known_list.append(feats)
            logger.info("Faces in database: %d", len(self.face_features_known_list))
            return True
        except pd.errors.EmptyDataError:
            logger.warning("'features_all.csv' is empty.")
            return False

    # ── matching ──────────────────────────────────────────────────────────

    @staticmethod
    def euclidean(f1, f2) -> float:
        a = np.array(f1, dtype=float)
        b = np.array(f2, dtype=float)
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def centroid_tracker(self) -> None:
        for i, curr_c in enumerate(self.current_frame_face_centroid_list):
            dists = [self.euclidean(curr_c, last_c)
                     for last_c in self.last_frame_face_centroid_list]
            self.current_frame_face_name_list[i] = \
                self.last_frame_face_name_list[dists.index(min(dists))]

    # ── FPS ───────────────────────────────────────────────────────────────

    def update_fps(self) -> None:
        now = time.time()
        dt  = now - self._frame_start
        self.fps = 1.0 / dt if dt > 0 else 0
        self._frame_start = now

    # ── texture analysis ──────────────────────────────────────────────────

    def analyze_face_texture(self, face_roi: np.ndarray):
        if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return 0.0, False
        try:
            small = cv2.resize(face_roi, (64, 64))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hog   = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
            desc  = hog.compute(gray)
            var   = float(np.var(desc))
            score = var * 1000
            return score, score > 0.5
        except Exception as e:
            logger.debug("Texture analysis error: %s", e)
            return 0.0, False

    # ── motion analysis ───────────────────────────────────────────────────

    def analyze_face_motion(self, face_id: str, shape):
        pts = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(36, 48)] +
            [(shape.part(30).x, shape.part(30).y)],
            dtype=float,
        )
        if face_id not in self.face_motion_history:
            self.face_motion_history[face_id] = deque(maxlen=MOTION_FRAMES)
            self.prev_landmarks[face_id] = pts
            return 0.0, False
        prev = self.prev_landmarks[face_id]
        motion = float(np.mean(np.linalg.norm(pts - prev, axis=1)))
        self.face_motion_history[face_id].append(motion)
        self.prev_landmarks[face_id] = pts

        if len(self.face_motion_history[face_id]) >= MOTION_FRAMES // 2:
            history   = list(self.face_motion_history[face_id])
            avg       = float(np.mean(history))
            variance  = float(np.var(history))
            is_loop   = variance < 0.001 and len(history) == MOTION_FRAMES
            is_real   = avg > MOTION_THRESHOLD and variance > 0.01 and not is_loop
            return avg, is_real
        return 0.0, False

    # ── challenge-response ────────────────────────────────────────────────

    def generate_challenge(self, face_id: str) -> str:
        if not CHALLENGE_ACTIVE:
            return "NONE"
        if face_id not in self.current_challenges:
            self.current_challenges[face_id]  = random.choice(CHALLENGE_TYPES)
            self.challenge_progress[face_id]  = 0
            self.challenge_complete[face_id]  = False
        return self.current_challenges[face_id]

    def detect_nod(self, face_id: str, shape) -> tuple[bool, float]:
        nose_tip = (shape.part(30).x, shape.part(30).y)
        key = f"{face_id}__nose_pos"
        if key not in self.__dict__:
            self.__dict__[key] = deque(maxlen=5)
            self.__dict__[f"{face_id}__init_nose"] = nose_tip
        self.__dict__[key].append(nose_tip)
        init = self.__dict__[f"{face_id}__init_nose"]
        diffs = [p[1] - init[1] for p in self.__dict__[key]]
        if len(diffs) >= 4:
            changes = sum(
                1 for i in range(1, len(diffs))
                if (diffs[i] > 0) != (diffs[i - 1] > 0)
            )
            movement = max(diffs) - min(diffs)
            return changes >= 1 and movement > 5, float(movement)
        return False, 0.0

    def check_challenge_response(self, face_id: str, shape) -> bool:
        challenge = self.generate_challenge(face_id)
        if challenge == "NONE":
            self.challenge_complete[face_id] = True
            return True
        if self.challenge_complete.get(face_id, False):
            return True

        progress = self.challenge_progress.get(face_id, 0)
        if challenge == "BLINK":
            blinked, _, _, _ = self.detect_blink(shape)
            if blinked:
                progress += 5
                self.blink_detected = False
        elif challenge == "NOD":
            nodded, _ = self.detect_nod(face_id, shape)
            if nodded:
                progress += 5

        self.challenge_progress[face_id] = progress
        if progress >= 10:
            self.challenge_complete[face_id] = True
            return True
        return False

    # ── liveness composite score ──────────────────────────────────────────

    def detect_liveness(self, face_id: str, img_rd: np.ndarray, d, shape):
        try:
            roi = img_rd[
                max(0, d.top()): min(img_rd.shape[0], d.bottom()),
                max(0, d.left()): min(img_rd.shape[1], d.right()),
            ]
            if roi.size == 0:
                return False, "Invalid face region"
        except Exception as e:
            return False, f"ROI error: {e}"

        if face_id not in self.liveness_scores:
            self.liveness_scores[face_id] = 0.0
            self.spoof_detected[face_id]  = False

        score         = 0.0
        spoof_reasons = []

        # 1. Challenge (20 pts)
        challenge_ok = self.check_challenge_response(face_id, shape)
        if challenge_ok:
            score += 20
        else:
            spoof_reasons.append("Challenge incomplete")

        # 2. Motion (max 40 pts)
        motion_val, is_real_motion = self.analyze_face_motion(face_id, shape)
        if is_real_motion:
            score += min(40.0, motion_val * 200)
        else:
            spoof_reasons.append("Unnatural motion")

        # 3. Texture (max 40 pts, checked every 5 frames to save CPU)
        if self.frame_cnt % 5 == 0:
            tex_score, is_real_tex = self.analyze_face_texture(roi)
            if is_real_tex:
                score += min(40.0, tex_score)
            else:
                spoof_reasons.append("Fake texture")

        # Exponential smoothing
        alpha = 0.30
        self.liveness_scores[face_id] = (
            alpha * score + (1 - alpha) * self.liveness_scores[face_id]
        )

        is_live = self.liveness_scores[face_id] >= 50
        if not is_live and self.frame_cnt > 30:
            self.spoof_detected[face_id] = True

        msg = ", ".join(spoof_reasons) if spoof_reasons else "Live"
        return is_live, msg

    # ── attendance ────────────────────────────────────────────────────────

    def attendance(self, name: str) -> None:
        is_live   = self.liveness_scores.get(name, 0) >= 50
        has_blink = self.person_blink_status.get(name, False)
        spoof     = self.spoof_detected.get(name, False)

        if self.blink_detected:
            self.person_blink_status[name] = True
            has_blink = True

        if spoof:
            status = "INVALID - SPOOFING DETECTED"
            print(f"⚠  {name}: spoofing detected — INVALID")
        elif BLINK_REQUIRED and not has_blink:
            status = "INVALID - No Blink Detected"
            print(f"⚠  {name}: no blink — INVALID")
        elif CHALLENGE_ACTIVE and not self.challenge_complete.get(name, False) and not is_live:
            status = "INVALID - Challenge Not Completed"
            print(f"⚠  {name}: challenge not completed — INVALID")
        elif not is_live:
            status = "INVALID - Failed Liveness Check"
            print(f"⚠  {name}: liveness failed — INVALID")
        else:
            status = "VALID"
            print(f"✓  {name}: VALID attendance")

        try:
            result = record_attendance(name, status)
            print(f"   DB: {result} — {status}")
        except Exception as e:
            logger.error("DB error recording attendance for %s: %s", name, e)

    # ── OSD helpers ───────────────────────────────────────────────────────

    def draw_note(self, img: np.ndarray) -> None:
        cv2.putText(img, "Face Recognition + Anti-Spoofing",
                    (10, 25), self.font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # ── main processing loop ──────────────────────────────────────────────

    def process(
        self,
        stream: cv2.VideoCapture,
        frame_callback,           # callable(np.ndarray RGB)
        metrics_callback,         # callable(dict, is_system_metrics=bool)
    ) -> None:
        if not self.get_face_database():
            return

        while stream.isOpened() and not getattr(self, "stop_thread", False):
            self.frame_cnt += 1
            ok, img_rd = stream.read()
            if not ok:
                break

            faces = detector(img_rd, 0)
            self.last_frame_face_cnt              = self.current_frame_face_cnt
            self.current_frame_face_cnt           = len(faces)
            self.last_frame_face_name_list        = self.current_frame_face_name_list[:]
            self.last_frame_face_centroid_list    = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []

            same_cnt  = self.current_frame_face_cnt == self.last_frame_face_cnt
            no_reclfy = self.reclassify_interval_cnt != self.reclassify_interval

            if same_cnt and no_reclfy:
                # ── scene 1: face count unchanged ────────────────────────────
                self.current_frame_face_position_list = []
                if "unknown" in self.current_frame_face_name_list:
                    self.reclassify_interval_cnt += 1

                for k, d in enumerate(faces):
                    self.current_frame_face_centroid_list.append(
                        [int((d.left() + d.right()) / 2), int((d.top() + d.bottom()) / 2)]
                    )
                    self.current_frame_face_position_list.append(
                        (d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4))
                    )

                    rect_color = (255, 255, 255)
                    face_metrics: dict = {}

                    name = (self.current_frame_face_name_list[k]
                            if k < len(self.current_frame_face_name_list) else "unknown")

                    if name != "unknown":
                        face_id = name
                        shape   = predictor(img_rd, d)
                        blinked, ear, lEye, rEye = self.detect_blink(shape)
                        is_live, spoof_msg = self.detect_liveness(face_id, img_rd, d, shape)
                        rect_color = (0, 255, 0) if is_live else (0, 0, 255)

                        if blinked:
                            self.person_blink_status[face_id] = True
                            self.blink_detected = False

                        # Eye contours
                        cv2.drawContours(img_rd, [np.array(lEye)], -1, (0, 255, 0), 1)
                        cv2.drawContours(img_rd, [np.array(rEye)], -1, (0, 255, 0), 1)

                        # Mouth aspect ratio
                        mouth = [(shape.part(i).x, shape.part(i).y) for i in range(48, 60)]
                        mar = 0.0
                        if mouth:
                            mw = dist.euclidean(mouth[0], mouth[6])
                            mh = dist.euclidean(mouth[3], mouth[9])
                            mar = mh / max(mw, 1e-6)

                        nose_len = dist.euclidean(
                            (shape.part(30).x, shape.part(30).y),
                            (shape.part(27).x, shape.part(27).y),
                        )
                        motion_val = float(np.mean(self.face_motion_history[face_id])) \
                            if face_id in self.face_motion_history \
                               and self.face_motion_history[face_id] else 0.0
                        liveness_val = self.liveness_scores.get(face_id, 0.0)
                        challenge    = self.current_challenges.get(face_id, "NONE")
                        progress     = self.challenge_progress.get(face_id, 0)

                        # OSD annotations
                        x0, y0, ls = d.right() + 10, d.top(), 22
                        cv2.putText(img_rd, f"ID: {face_id}",           (x0, y0),        self.font, 0.65, rect_color, 1)
                        cv2.putText(img_rd, f"Live: {liveness_val:.1f}",(x0, y0+ls),     self.font, 0.55, rect_color, 1)
                        cv2.putText(img_rd, f"Blink: {self.person_blink_status.get(face_id,False)}", (x0, y0+ls*2), self.font, 0.55, (0,255,0) if self.person_blink_status.get(face_id) else (0,0,255), 1)
                        cv2.putText(img_rd, f"EAR: {ear:.2f}",          (x0, y0+ls*3),   self.font, 0.55, (0,255,255), 1)
                        cv2.putText(img_rd, f"MAR: {mar:.2f}",          (x0, y0+ls*4),   self.font, 0.55, (0,255,255), 1)
                        cv2.putText(img_rd, f"Motion: {motion_val:.2f}",(x0, y0+ls*5),   self.font, 0.55, (0,255,255), 1)
                        if CHALLENGE_ACTIVE:
                            cc = (0,255,0) if self.challenge_complete.get(face_id) else (0,165,255)
                            cv2.putText(img_rd, f"Challenge: {challenge} ({progress}/10)",
                                        (x0, y0+ls*6), self.font, 0.55, cc, 1)
                        if not is_live:
                            cv2.putText(img_rd, f"SPOOF: {spoof_msg}",
                                        (d.left(), d.bottom() + 25), self.font, 0.65, (0,0,255), 1)

                        # Key landmark dots
                        for i in [0, 8, 16, 27, 30, 36, 39, 42, 45, 48, 54]:
                            cv2.circle(img_rd, (shape.part(i).x, shape.part(i).y), 2, (0,255,255), -1)

                        face_metrics = {
                            "face_id":     face_id,
                            "liveness":    liveness_val,
                            "blinked":     self.person_blink_status.get(face_id, False),
                            "ear":         ear,
                            "mar":         mar,
                            "motion":      motion_val,
                            "nose_length": nose_len,
                            "challenge":   f"{challenge} ({progress}/10)",
                            "spoof_message": spoof_msg if not is_live else "Live",
                        }

                    cv2.rectangle(img_rd, (d.left(), d.top()), (d.right(), d.bottom()), rect_color, 2)
                    metrics_callback(face_metrics)

                if self.current_frame_face_cnt > 1:
                    self.centroid_tracker()
                for i in range(self.current_frame_face_cnt):
                    cv2.putText(img_rd,
                                self.current_frame_face_name_list[i],
                                self.current_frame_face_position_list[i],
                                self.font, 0.8, (0, 255, 255), 1)

            else:
                # ── scene 2: face count changed ───────────────────────────────
                self.current_frame_face_position_list     = []
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_feature_list      = []
                self.reclassify_interval_cnt = 0

                if self.current_frame_face_cnt == 0:
                    self.current_frame_face_name_list = []
                    metrics_callback({})
                else:
                    self.current_frame_face_name_list = []
                    for i, face in enumerate(faces):
                        shape = predictor(img_rd, face)
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(img_rd, shape)
                        )
                        self.current_frame_face_name_list.append("unknown")
                        blinked, ear, lEye, rEye = self.detect_blink(shape)
                        cv2.drawContours(img_rd, [np.array(lEye)], -1, (0,255,0), 1)
                        cv2.drawContours(img_rd, [np.array(rEye)], -1, (0,255,0), 1)
                        cv2.putText(img_rd, f"EAR:{ear:.2f}",
                                    (face.right()+10, face.top()), self.font, 0.65, (0,255,255), 1)

                    for k, face in enumerate(faces):
                        cx = int((face.left() + face.right()) / 2)
                        cy = int((face.top()  + face.bottom()) / 2)
                        self.current_frame_face_centroid_list.append([cx, cy])
                        self.current_frame_face_position_list.append(
                            (face.left(), int(face.bottom() + (face.bottom()-face.top())/4))
                        )

                        dists_list = []
                        for feat in self.face_features_known_list:
                            d_val = self.euclidean(self.current_frame_face_feature_list[k], feat) \
                                if str(feat[0]) != "0.0" else 999999
                            dists_list.append(d_val)

                        best_idx  = dists_list.index(min(dists_list))
                        best_dist = min(dists_list)
                        rect_color = (255, 255, 255)

                        if best_dist < 0.4:
                            face_id = self.face_name_known_list[best_idx]
                            self.current_frame_face_name_list[k] = face_id
                            shape = predictor(img_rd, face)
                            is_live, spoof_msg = self.detect_liveness(face_id, img_rd, face, shape)
                            rect_color = (0, 255, 0) if is_live else (0, 0, 255)
                            liveness_val = self.liveness_scores.get(face_id, 0.0)
                            cv2.putText(img_rd, f"Live:{liveness_val:.1f}",
                                        (face.right()+10, face.top()+25), self.font, 0.65, rect_color, 1)
                            if not is_live:
                                cv2.putText(img_rd, f"SPOOF:{spoof_msg}",
                                            (face.left(), face.bottom()+25), self.font, 0.65, (0,0,255), 1)
                            self.attendance(face_id)
                            metrics_callback({
                                "face_id":     face_id,
                                "liveness":    liveness_val,
                                "blinked":     self.person_blink_status.get(face_id, False),
                                "ear": 0.0, "mar": 0.0, "motion": 0.0, "nose_length": 0.0,
                                "challenge":   f"{self.current_challenges.get(face_id,'NONE')} ({self.challenge_progress.get(face_id,0)}/10)",
                                "spoof_message": spoof_msg if not is_live else "Live",
                            })

                        cv2.rectangle(img_rd, (face.left(), face.top()),
                                      (face.right(), face.bottom()), rect_color, 2)

            self.draw_note(img_rd)
            self.update_fps()

            # ── convert BGR→RGB and emit to UI ────────────────────────────
            img_rgb = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
            frame_callback(img_rgb)
            metrics_callback(
                {
                    "fps":       self.fps,
                    "frame_cnt": self.frame_cnt,
                    "faces":     self.current_frame_face_cnt,
                    "blinks":    self.total_blinks,
                },
                is_system_metrics=True,
            )

    def run(self, cap: cv2.VideoCapture, frame_callback, metrics_callback) -> None:
        """Process frames from an already-opened VideoCapture (opened on main thread)."""
        for _ in range(5):
            cap.read()   # flush initial frames
        self.process(cap, frame_callback, metrics_callback)


# ===========================================================================
# Background QThread for video processing
# ===========================================================================

class VideoThread(QThread):
    frame_signal   = pyqtSignal(np.ndarray)   # RGB frame
    metrics_signal = pyqtSignal(dict, bool)   # (metrics, is_system)

    def __init__(self, face_recognizer: Face_Recognizer, cap: cv2.VideoCapture) -> None:
        super().__init__()
        self.fr  = face_recognizer
        self.cap = cap  # camera opened on main thread (macOS AVFoundation requirement)

    def run(self) -> None:
        self.fr.run(self.cap, self._on_frame, self._on_metrics)

    def _on_frame(self, rgb: np.ndarray) -> None:
        self.frame_signal.emit(rgb)

    def _on_metrics(self, metrics: dict, is_system_metrics: bool = False) -> None:
        self.metrics_signal.emit(metrics, is_system_metrics)

    def stop(self) -> None:
        self.fr.stop_thread = True
        self.wait(3000)


# ===========================================================================
# PyQt6 main application window
# ===========================================================================

class FaceRecognitionApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Face Recognition  |  Anti-Spoofing Attendance System")
        self.setMinimumSize(1100, 660)

        self.face_recognizer = Face_Recognizer()
        self.cap = self._open_camera()   # MUST open on main thread (macOS AVFoundation)
        self._build_ui()
        self._start_video_thread()

    # =========================================================================
    # Camera init (must run on main thread on macOS)
    # =========================================================================

    def _open_camera(self) -> cv2.VideoCapture | None:
        """Open the camera on the main thread so macOS AVFoundation can request permission."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None, "Camera Error",
                "Could not open camera.\n"
                "• Check macOS Privacy → Camera permissions for Terminal / Python.\n"
                "• No other app should be using the camera.",
            )
            logger.error("Could not open camera — check macOS camera permissions")
            return None
        logger.info("Camera opened on main thread ✓")
        return cap

    # =========================================================================
    # UI
    # =========================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Face Recognition  ·  Anti-Spoofing Attendance")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
        root.addWidget(title)

        # Content row
        content = QHBoxLayout()
        content.setSpacing(10)

        # ── video display ─────────────────────────────────────────────────
        vid_group = QGroupBox("Live Camera")
        vid_layout = QVBoxLayout(vid_group)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background:#111;")
        vid_layout.addWidget(self.video_label)
        content.addWidget(vid_group, stretch=3)

        # ── metrics panel ────────────────────────────────────────────────
        metrics_widget = QWidget()
        metrics_widget.setMaximumWidth(330)
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setSpacing(6)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        metrics_layout.addWidget(self._build_system_info_panel())
        metrics_layout.addWidget(self._build_face_metrics_panel())
        metrics_layout.addWidget(self._build_antispoofing_panel())
        metrics_layout.addStretch()

        content.addWidget(metrics_widget, stretch=1)
        root.addLayout(content, stretch=1)

        # Quit
        quit_btn = QPushButton("✖  Quit")
        quit_btn.setStyleSheet(
            "background:#e53935; color:white; padding:10px; font-size:13px;"
            "font-weight:bold; border-radius:4px;"
        )
        quit_btn.clicked.connect(self._quit)
        root.addWidget(quit_btn)

    def _build_system_info_panel(self) -> QGroupBox:
        group = QGroupBox("System Info")
        grid  = QGridLayout(group)
        grid.setColumnStretch(1, 1)

        def row(label, row_idx):
            grid.addWidget(QLabel(label), row_idx, 0)
            lbl = QLabel("—")
            grid.addWidget(lbl, row_idx, 1)
            return lbl

        self.lbl_fps    = row("FPS:",    0)
        self.lbl_frame  = row("Frame:",  1)
        self.lbl_faces  = row("Faces:",  2)
        self.lbl_blinks = row("Blinks:", 3)
        return group

    def _build_face_metrics_panel(self) -> QGroupBox:
        group = QGroupBox("Face Metrics")
        grid  = QGridLayout(group)
        grid.setColumnStretch(1, 1)

        def row(label, row_idx):
            grid.addWidget(QLabel(label), row_idx, 0)
            lbl = QLabel("—")
            grid.addWidget(lbl, row_idx, 1)
            return lbl

        self.lbl_face_id    = row("Face ID:",     0)
        self.lbl_liveness   = row("Liveness:",    1)
        self.lbl_blinked    = row("Blinked:",     2)
        self.lbl_ear        = row("EAR:",         3)
        self.lbl_mar        = row("MAR:",         4)
        self.lbl_motion     = row("Motion:",      5)
        self.lbl_nose       = row("Nose Length:", 6)
        self.lbl_challenge  = row("Challenge:",   7)
        self.lbl_spoof      = row("Spoof Status:",8)
        return group

    def _build_antispoofing_panel(self) -> QGroupBox:
        group  = QGroupBox("Anti-Spoofing Methods Active")
        layout = QVBoxLayout(group)
        for method in [
            "✅  Eye Blink Detection (EAR)",
            "✅  Motion Analysis (landmark variance)",
            "✅  Texture Analysis (HOG descriptor)",
            "✅  Challenge-Response (Blink / Nod)",
            "✅  Composite Liveness Score (≥50 = VALID)",
        ]:
            lbl = QLabel(method)
            lbl.setFont(QFont("", 9))
            layout.addWidget(lbl)
        return group

    # =========================================================================
    # Video thread
    # =========================================================================

    def _start_video_thread(self) -> None:
        if not self.cap:
            return  # camera failed to open
        self.video_thread = VideoThread(self.face_recognizer, self.cap)
        self.video_thread.frame_signal.connect(self._update_frame)
        self.video_thread.metrics_signal.connect(self._update_metrics)
        self.video_thread.start()

    # =========================================================================
    # Slots
    # =========================================================================

    def _update_frame(self, rgb: np.ndarray) -> None:
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_metrics(self, metrics: dict, is_system: bool) -> None:
        if is_system:
            self.lbl_fps.setText(f"{metrics.get('fps', 0):.2f}")
            self.lbl_frame.setText(str(metrics.get("frame_cnt", 0)))
            self.lbl_faces.setText(str(metrics.get("faces", 0)))
            self.lbl_blinks.setText(str(metrics.get("blinks", 0)))
            return

        if not metrics:
            for lbl in [self.lbl_face_id, self.lbl_liveness, self.lbl_blinked,
                        self.lbl_ear, self.lbl_mar, self.lbl_motion,
                        self.lbl_nose, self.lbl_challenge, self.lbl_spoof]:
                lbl.setText("—")
            return

        liveness = metrics.get("liveness", 0.0)
        self.lbl_face_id.setText(str(metrics.get("face_id", "—")))
        self.lbl_liveness.setText(f"{liveness:.1f}")
        self.lbl_liveness.setStyleSheet(
            "color:green; font-weight:bold;" if liveness >= 50 else "color:red;"
        )
        blinked = metrics.get("blinked", False)
        self.lbl_blinked.setText(str(blinked))
        self.lbl_blinked.setStyleSheet(
            "color:green;" if blinked else "color:orange;"
        )
        self.lbl_ear.setText(f"{metrics.get('ear', 0):.3f}")
        self.lbl_mar.setText(f"{metrics.get('mar', 0):.3f}")
        self.lbl_motion.setText(f"{metrics.get('motion', 0):.3f}")
        self.lbl_nose.setText(f"{metrics.get('nose_length', 0):.1f}")
        self.lbl_challenge.setText(str(metrics.get("challenge", "NONE")))
        spoof_msg = str(metrics.get("spoof_message", "—"))
        self.lbl_spoof.setText(spoof_msg)
        self.lbl_spoof.setStyleSheet(
            "color:green;" if spoof_msg in ("Live", "No spoof detected") else "color:red;"
        )

    def _quit(self) -> None:
        self.video_thread.stop()
        self.close()

    def closeEvent(self, event) -> None:
        if hasattr(self, "video_thread"):
            self.video_thread.stop()
        if self.cap:
            self.cap.release()
        event.accept()


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    # Initialise PostgreSQL schema before starting the GUI
    try:
        init_db()
    except Exception as e:
        logger.error("Cannot connect to database: %s", e)
        print(f"\n❌  Database connection failed: {e}")
        print("    Make sure PostgreSQL is running:  docker compose up -d\n")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FaceRecognitionApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
