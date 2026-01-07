import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import random
from scipy.spatial import distance as dist
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Dlib / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Define eye landmarks indices
(LEFT_EYE_START, LEFT_EYE_END) = (42, 48)
(RIGHT_EYE_START, RIGHT_EYE_END) = (36, 42)

# Constants for blink detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 1
BLINK_REQUIRED = True

# Anti-spoofing constants
FACIAL_LANDMARKS_IDXS = {
    "mouth": (48, 68),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "nose": (27, 35),
    "jaw": (0, 17)
}
LBP_POINTS = 24
LBP_RADIUS = 3
MOTION_FRAMES = 10
MOTION_THRESHOLD = 0.05
CHALLENGE_ACTIVE = True
CHALLENGE_TYPES = ["BLINK", "NOD"]
CHALLENGE_DURATION = 50

# Create a connection to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Check if attendance table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance'")
table_exists = cursor.fetchone()

if table_exists:
    cursor.execute("PRAGMA table_info(attendance)")
    columns = cursor.fetchall()
    status_column_exists = any(column[1] == 'status' for column in columns)
    if not status_column_exists:
        cursor.execute("ALTER TABLE attendance ADD COLUMN status TEXT")
        print("Added 'status' column to existing attendance table")
else:
    current_date = datetime.datetime.now().strftime("%Y_%m_%d")
    table_name = "attendance"
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, status TEXT, UNIQUE(name, date))"
    cursor.execute(create_table_sql)

conn.commit()
conn.close()


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        self.frame_cnt = 0
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10
        self.eye_counter = 0
        self.total_blinks = 0
        self.blink_detected = False
        self.person_blink_status = {}
        self.face_motion_history = {}
        self.liveness_scores = {}
        self.spoof_detected = {}
        self.current_challenges = {}
        self.challenge_progress = {}
        self.challenge_complete = {}
        self.prev_landmarks = {}

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def get_eyes(self, shape):
        leftEye = []
        rightEye = []
        for i in range(LEFT_EYE_START, LEFT_EYE_END):
            leftEye.append((shape.part(i).x, shape.part(i).y))
        for i in range(RIGHT_EYE_START, RIGHT_EYE_END):
            rightEye.append((shape.part(i).x, shape.part(i).y))
        return leftEye, rightEye

    def detect_blink(self, shape):
        leftEye, rightEye = self.get_eyes(shape)
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        if ear < EYE_AR_THRESH:
            self.eye_counter += 1
            if self.eye_counter >= EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
                self.blink_detected = True
        else:
            if self.eye_counter > 0:
                self.eye_counter = 0
        return self.blink_detected, ear, leftEye, rightEye

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            return 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time if self.frame_time > 0 else 0
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)
            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognition", (20, 40), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        is_live = self.liveness_scores.get(name, 0) >= 50
        has_blinked = self.person_blink_status.get(name, False)
        if self.blink_detected:
            self.person_blink_status[name] = True
            has_blinked = True
            print(f"Blink detected for {name}!")
        spoof_detected = self.spoof_detected.get(name, False)
        if spoof_detected:
            status = "INVALID - SPOOFING DETECTED"
            print(f"WARNING: {name} attempted to use a fake face! Marking as INVALID.")
        elif BLINK_REQUIRED and not has_blinked:
            status = "INVALID - User is NOT Blinking the EYES."
            print(f"WARNING: {name} tried to mark attendance without blinking!")
        elif CHALLENGE_ACTIVE and not self.challenge_complete.get(name, False):
            status = "INVALID - Challenge Not Completed"
            print(f"WARNING: {name} did not complete the verification challenge!")
        elif not is_live:
            status = "INVALID - Failed Liveness Check"
            print(f"WARNING: {name} failed the liveness check!")
        else:
            status = "VALID"
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_entry = cursor.fetchone()
        if existing_entry:
            existing_status = existing_entry[3] if len(existing_entry) > 3 and existing_entry[3] is not None else "UNKNOWN"
            if (existing_status != "VALID" and status == "VALID") or has_blinked:
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                if has_blinked:
                    status = "VALID"
                cursor.execute("UPDATE attendance SET time = ?, status = ? WHERE name = ? AND date = ?",
                              (current_time, status, name, current_date))
                conn.commit()
                print(f"{name} attendance updated to {status} for {current_date} at {current_time}")
            else:
                print(f"{name} is already marked as present for {current_date}")
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, time, date, status) VALUES (?, ?, ?, ?)",
                          (name, current_time, current_date, status))
            conn.commit()
            print(f"{name} marked as present for {current_date} at {current_time} - Status: {status}")
        conn.close()

    def get_texture_features(self, image):
        small_img = cv2.resize(image, (64, 64))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        h = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        descriptors = h.compute(gray)
        return descriptors

    def analyze_face_texture(self, image):
        if image.size == 0 or image.shape[0] < 20 or image.shape[1] < 20:
            return 0, False
        try:
            descriptors = self.get_texture_features(image)
            texture_variance = np.var(descriptors)
            texture_score = texture_variance * 1000
            is_real = texture_score > 0.5
            return texture_score, is_real
        except Exception as e:
            logging.error(f"Texture analysis error: {e}")
            return 0, False

    def analyze_face_motion(self, face_id, shape):
        landmarks = []
        for i in range(36, 48):
            landmarks.append((shape.part(i).x, shape.part(i).y))
        landmarks.append((shape.part(30).x, shape.part(30).y))
        landmarks = np.array(landmarks)
        if face_id not in self.face_motion_history:
            self.face_motion_history[face_id] = deque(maxlen=MOTION_FRAMES)
            self.prev_landmarks[face_id] = landmarks
            return 0, False
        if face_id in self.prev_landmarks:
            motion = np.mean(np.linalg.norm(landmarks - self.prev_landmarks[face_id], axis=1))
            self.face_motion_history[face_id].append(motion)
            self.prev_landmarks[face_id] = landmarks
            if len(self.face_motion_history[face_id]) >= MOTION_FRAMES // 2:
                avg_motion = np.mean(self.face_motion_history[face_id])
                motion_variance = np.var(self.face_motion_history[face_id])
                is_real_motion = avg_motion > MOTION_THRESHOLD and motion_variance > 0.01
                is_video_loop = motion_variance < 0.001 and len(self.face_motion_history[face_id]) == MOTION_FRAMES
                return avg_motion, is_real_motion and not is_video_loop
        return 0, False

    def generate_challenge(self, face_id):
        if not CHALLENGE_ACTIVE:
            return "NONE"
        if face_id not in self.current_challenges:
            challenge = random.choice(CHALLENGE_TYPES)
            self.current_challenges[face_id] = challenge
            self.challenge_progress[face_id] = 0
            self.challenge_complete[face_id] = False
            return challenge
        return self.current_challenges[face_id]

    def detect_nod(self, face_id, shape):
        nose_tip = (shape.part(30).x, shape.part(30).y)
        if f"{face_id}_nose_positions" not in self.__dict__:
            self.__dict__[f"{face_id}_nose_positions"] = deque(maxlen=5)
            self.__dict__[f"{face_id}_nose_positions"].append(nose_tip)
            self.__dict__[f"{face_id}_nod_detected"] = False
            self.__dict__[f"{face_id}_initial_nose_pos"] = nose_tip
            return False, 0
        self.__dict__[f"{face_id}_nose_positions"].append(nose_tip)
        initial_pos = self.__dict__[f"{face_id}_initial_nose_pos"]
        positions = self.__dict__[f"{face_id}_nose_positions"]
        vert_diffs = [pos[1] - initial_pos[1] for pos in positions]
        if len(vert_diffs) >= 4:
            direction_changes = 0
            for i in range(1, len(vert_diffs)):
                if (vert_diffs[i] > 0 and vert_diffs[i-1] < 0) or (vert_diffs[i] < 0 and vert_diffs[i-1] > 0):
                    direction_changes += 1
            max_movement = max(vert_diffs) - min(vert_diffs)
            self.__dict__[f"{face_id}_nod_detected"] = direction_changes >= 1 and max_movement > 5
            return self.__dict__[f"{face_id}_nod_detected"], max_movement
        return False, 0

    def check_challenge_response(self, face_id, shape):
        challenge = self.current_challenges.get(face_id, "NONE")
        if challenge == "NONE":
            self.challenge_complete[face_id] = True
            return True
        progress = self.challenge_progress.get(face_id, 0)
        if challenge == "BLINK":
            blink_detected, _, _, _ = self.detect_blink(shape)
            if blink_detected:
                progress += 5
                self.blink_detected = False
        elif challenge == "NOD":
            nod_detected, _ = self.detect_nod(face_id, shape)
            if nod_detected:
                progress += 5
        self.challenge_progress[face_id] = progress
        if progress >= 10:
            self.challenge_complete[face_id] = True
            return True
        return False

    def detect_liveness(self, face_id, img_rd, d, shape):
        try:
            face_roi = img_rd[max(0, d.top()):min(img_rd.shape[0], d.bottom()),
                             max(0, d.left()):min(img_rd.shape[1], d.right())]
            if face_roi.size == 0:
                return False, "Invalid face region"
        except Exception as e:
            logging.error(f"Face ROI extraction error: {e}")
            return False, "Face detection error"
        if face_id not in self.liveness_scores:
            self.liveness_scores[face_id] = 0
            self.spoof_detected[face_id] = False
        liveness_score = 0
        spoof_reasons = []
        challenge_complete = self.check_challenge_response(face_id, shape)
        if challenge_complete:
            liveness_score += 40
        else:
            spoof_reasons.append(f"Challenge not completed")
        if liveness_score < 40:
            motion_score, is_real_motion = self.analyze_face_motion(face_id, shape)
            if is_real_motion:
                liveness_score += min(30, motion_score * 200)
            else:
                spoof_reasons.append("Unnatural motion")
            if self.frame_cnt % 5 == 0:
                texture_score, is_real_texture = self.analyze_face_texture(face_roi)
                if is_real_texture:
                    liveness_score += min(30, texture_score)
                else:
                    spoof_reasons.append("Fake texture detected")
        alpha = 0.3
        self.liveness_scores[face_id] = alpha * liveness_score + (1 - alpha) * self.liveness_scores.get(face_id, 0)
        is_live = self.liveness_scores[face_id] >= 50
        if not is_live and self.frame_cnt > 30:
            self.spoof_detected[face_id] = True
        spoof_message = ", ".join(spoof_reasons) if spoof_reasons else "No spoof detected"
        return is_live, spoof_message

    def process(self, stream, video_label, update_metrics_callback):
        if self.get_face_database():
            while stream.isOpened() and not hasattr(self, 'stop_thread'):
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                if not flag:
                    break
                faces = detector(img_rd, 0)
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1: No face cnt changes in this frame!!!")
                    self.current_frame_face_position_list = []
                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1
                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(d.left() + d.right()) / 2, int(d.top() + d.bottom()) / 2])
                            rect_color = (255, 255, 255)
                            metrics = {}
                            if self.current_frame_face_name_list[k] != "unknown":
                                face_id = self.current_frame_face_name_list[k]
                                shape = predictor(img_rd, d)
                                blink_detected, ear, leftEye, rightEye = self.detect_blink(shape)
                                is_live, spoof_message = self.detect_liveness(face_id, img_rd, d, shape)
                                if is_live:
                                    rect_color = (0, 255, 0)
                                elif self.spoof_detected.get(face_id, False):
                                    rect_color = (0, 0, 255)
                                challenge = self.current_challenges.get(face_id, "NONE")
                                progress = self.challenge_progress.get(face_id, 0)
                                leftEyeHull = np.array(leftEye)
                                rightEyeHull = np.array(rightEye)
                                cv2.drawContours(img_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                                cv2.drawContours(img_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                                mouth_pts = []
                                for i in range(48, 60):
                                    mouth_pts.append((shape.part(i).x, shape.part(i).y))
                                mouth_pts = np.array(mouth_pts)
                                mar = 0
                                if len(mouth_pts) > 0:
                                    mouth_width = dist.euclidean(mouth_pts[0], mouth_pts[6])
                                    mouth_height = dist.euclidean(mouth_pts[3], mouth_pts[9])
                                    mar = mouth_height / max(mouth_width, 1e-6)
                                nose_tip = (shape.part(30).x, shape.part(30).y)
                                nose_bridge = (shape.part(27).x, shape.part(27).y)
                                nose_length = dist.euclidean(nose_tip, nose_bridge)
                                x_pos = d.right() + 10
                                y_pos_start = d.top()
                                line_space = 22
                                cv2.putText(img_rd, f"Face ID: {face_id}", (x_pos, y_pos_start),
                                            self.font, 0.7, rect_color, 1, cv2.LINE_AA)
                                liveness_score = self.liveness_scores.get(face_id, 0)
                                cv2.putText(img_rd, f"Liveness: {liveness_score:.1f}%",
                                            (x_pos, y_pos_start + line_space * 1),
                                            self.font, 0.6, rect_color, 1, cv2.LINE_AA)
                                if blink_detected:
                                    self.person_blink_status[face_id] = True
                                    self.blink_detected = False
                                blink_status = self.person_blink_status.get(face_id, False)
                                status_color = (0, 255, 0) if blink_status else (0, 0, 255)
                                cv2.putText(img_rd, f"Blinked: {blink_status}",
                                            (x_pos, y_pos_start + line_space * 2),
                                            self.font, 0.6, status_color, 1, cv2.LINE_AA)
                                cv2.putText(img_rd, f"EAR: {ear:.2f}",
                                            (x_pos, y_pos_start + line_space * 3),
                                            self.font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(img_rd, f"MAR: {mar:.2f}",
                                            (x_pos, y_pos_start + line_space * 4),
                                            self.font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                                motion_score = 0
                                if face_id in self.face_motion_history and len(self.face_motion_history[face_id]) > 0:
                                    motion_score = np.mean(self.face_motion_history[face_id])
                                cv2.putText(img_rd, f"Motion: {motion_score:.2f}",
                                            (x_pos, y_pos_start + line_space * 5),
                                            self.font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(img_rd, f"Nose L: {nose_length:.2f}",
                                            (x_pos, y_pos_start + line_space * 6),
                                            self.font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                                if CHALLENGE_ACTIVE:
                                    challenge_color = (0, 255, 0) if self.challenge_complete.get(face_id, False) else (0, 165, 255)
                                    cv2.putText(img_rd, f"Challenge: {challenge} ({progress}/10)",
                                                (x_pos, y_pos_start + line_space * 7),
                                                self.font, 0.6, challenge_color, 1, cv2.LINE_AA)
                                if not is_live:
                                    cv2.putText(img_rd, f"ALERT: {spoof_message}",
                                                (d.left(), d.bottom() + 25),
                                                self.font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                                key_landmarks = [0, 8, 16, 27, 30, 36, 39, 42, 45, 48, 54]
                                for i in key_landmarks:
                                    point = (shape.part(i).x, shape.part(i).y)
                                    cv2.circle(img_rd, point, 2, (0, 255, 255), -1)
                                metrics = {
                                    'face_id': face_id,
                                    'liveness': liveness_score,
                                    'blinked': blink_status,
                                    'ear': ear,
                                    'mar': mar,
                                    'motion': motion_score,
                                    'nose_length': nose_length,
                                    'challenge': f"{challenge} ({progress}/10)",
                                    'spoof_message': spoof_message if not is_live else "No spoof detected"
                                }
                            img_rd = cv2.rectangle(img_rd, tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]), rect_color, 2)
                            update_metrics_callback(metrics)
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()
                    for i in range(self.current_frame_face_cnt):
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        self.current_frame_face_name_list = []
                        update_metrics_callback({})
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")
                            blink_detected, ear, leftEye, rightEye = self.detect_blink(shape)
                            leftEyeHull = np.array(leftEye)
                            rightEyeHull = np.array(rightEye)
                            cv2.drawContours(img_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                            cv2.drawContours(img_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                            cv2.putText(img_rd, f"EAR: {ear:.2f}",
                                        (faces[i].right() + 10, faces[i].top()),
                                        self.font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                        for k in range(len(faces)):
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_X_e_distance_list = []
                            for i in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    self.current_frame_face_X_e_distance_list.append(999999999)
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))
                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                face_id = self.face_name_known_list[similar_person_num]
                                shape = predictor(img_rd, faces[k])
                                is_live, spoof_message = self.detect_liveness(face_id, img_rd, faces[k], shape)
                                rect_color = (0, 255, 0) if is_live else (0, 0, 255)
                                img_rd = cv2.rectangle(img_rd,
                                                       tuple([faces[k].left(), faces[k].top()]),
                                                       tuple([faces[k].right(), faces[k].bottom()]),
                                                       rect_color, 2)
                                liveness_score = self.liveness_scores.get(face_id, 0)
                                cv2.putText(img_rd, f"Liveness: {liveness_score:.1f}%",
                                            (faces[k].right() + 10, faces[k].top() + 100),
                                            self.font, 0.7, rect_color, 1, cv2.LINE_AA)
                                if not is_live:
                                    cv2.putText(img_rd, f"ALERT: {spoof_message}",
                                                (faces[k].left(), faces[k].bottom() + 25),
                                                self.font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                                self.attendance(face_id)
                                metrics = {
                                    'face_id': face_id,
                                    'liveness': liveness_score,
                                    'blinked': self.person_blink_status.get(face_id, False),
                                    'ear': 0,
                                    'mar': 0,
                                    'motion': 0,
                                    'nose_length': 0,
                                    'challenge': f"{self.current_challenges.get(face_id, 'NONE')} ({self.challenge_progress.get(face_id, 0)}/10)",
                                    'spoof_message': spoof_message if not is_live else "No spoof detected"
                                }
                                update_metrics_callback(metrics)
                        self.draw_note(img_rd)
                self.update_fps()
                # Resize image for display
                img_rd = cv2.resize(img_rd, (320, 240))
                img_rgb = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                video_label.img_tk = img_tk
                video_label.configure(image=img_tk)
                update_metrics_callback({
                    'fps': self.fps,
                    'frame_cnt': self.frame_cnt,
                    'faces': self.current_frame_face_cnt,
                    'blinks': self.total_blinks
                }, is_system_metrics=True)
                video_label.update()

    def run(self, video_label, update_metrics_callback):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            logging.error("Error: Could not open camera")
            return
        for _ in range(10):
            cap.read()
        self.process(cap, video_label, update_metrics_callback)
        cap.release()


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.configure(bg="#F0F0F0")
        self.face_recognizer = Face_Recognizer()
        self.running = True
        self.setup_ui()
        self.start_processing()

    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg="#FFFFFF", padx=300, pady=150)
        self.main_frame.pack(expand=True, fill="both")

        # Title
        tk.Label(self.main_frame, text="Face Recognition with Anti-Spoofing", font=("Helvetica", 18, "bold"),
                 bg="#FFFFFF", fg="#000000").pack(pady=10)
        # Video and metrics container
        self.content_frame = tk.Frame(self.main_frame, bg="#FFFFFF")
        self.content_frame.pack(fill="both", expand=True)

        # Video frame
        self.video_frame = tk.Frame(self.content_frame, bg="#FFFFFF", bd=2, relief="solid", width=626, height=417)
        self.video_frame.pack(side="left", padx=10, pady=10)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg="#FFFFFF")
        self.video_label.pack()

        # Metrics frame
        self.metrics_frame = tk.Frame(self.content_frame, bg="#FFFFFF")
        self.metrics_frame.pack(side="left", padx=10, pady=10, fill="both")

        # System Info
        self.system_frame = tk.LabelFrame(self.metrics_frame, text="System Info", font=("Helvetica", 12, "bold"),
                                          bg="#F0F0F0", fg="#333333", padx=10, pady=5)
        self.system_frame.pack(fill="x", pady=5)
        self.fps_label = tk.Label(self.system_frame, text="FPS: 0.0", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.fps_label.pack(anchor="w")
        self.frame_label = tk.Label(self.system_frame, text="Frame: 0", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.frame_label.pack(anchor="w")
        self.faces_label = tk.Label(self.system_frame, text="Faces: 0", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.faces_label.pack(anchor="w")
        self.blinks_label = tk.Label(self.system_frame, text="Blinks: 0", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.blinks_label.pack(anchor="w")

        # Face Metrics
        self.face_frame = tk.LabelFrame(self.metrics_frame, text="Face Metrics", font=("Helvetica", 12, "bold"),
                                        bg="#F0F0F0", fg="#333333", padx=10, pady=5)
        self.face_frame.pack(fill="x", pady=5)
        self.face_id_label = tk.Label(self.face_frame, text="Face ID: None", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.face_id_label.pack(anchor="w")
        self.liveness_label = tk.Label(self.face_frame, text="Liveness: 0.0%", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.liveness_label.pack(anchor="w")
        self.blinked_label = tk.Label(self.face_frame, text="Blinked: False", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.blinked_label.pack(anchor="w")
        self.ear_label = tk.Label(self.face_frame, text="EAR: 0.00", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.ear_label.pack(anchor="w")
        self.mar_label = tk.Label(self.face_frame, text="MAR: 0.00", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.mar_label.pack(anchor="w")
        self.motion_label = tk.Label(self.face_frame, text="Motion: 0.00", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.motion_label.pack(anchor="w")
        self.nose_label = tk.Label(self.face_frame, text="Nose Length: 0.00", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.nose_label.pack(anchor="w")
        self.challenge_label = tk.Label(self.face_frame, text="Challenge: NONE", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.challenge_label.pack(anchor="w")
        self.spoof_label = tk.Label(self.face_frame, text="Spoof: None", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333")
        self.spoof_label.pack(anchor="w")

        # Anti-Spoofing Methods
        self.spoof_frame = tk.LabelFrame(self.metrics_frame, text="Anti-Spoofing Methods", font=("Helvetica", 12, "bold"),
                                         bg="#F0F0F0", fg="#333333", padx=10, pady=5)
        self.spoof_frame.pack(fill="x", pady=5)
        tk.Label(self.spoof_frame, text="- Texture Analysis", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333").pack(anchor="w")
        tk.Label(self.spoof_frame, text="- Motion Analysis", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333").pack(anchor="w")
        tk.Label(self.spoof_frame, text="- Challenge-Response", font=("Helvetica", 10), bg="#F0F0F0", fg="#333333").pack(anchor="w")

        # Quit Button
        self.quit_button = tk.Button(self.main_frame, text="Quit", font=("Helvetica", 12, "bold"), bg="#FF4C4C", fg="#FFFFFF",command=self.quit, width=10)
        self.quit_button.pack(pady=10)      

    def update_metrics(self, metrics, is_system_metrics=False):
        if is_system_metrics:
            self.fps_label.configure(text=f"FPS: {metrics.get('fps', 0.0):.2f}")
            self.frame_label.configure(text=f"Frame: {metrics.get('frame_cnt', 0)}")
            self.faces_label.configure(text=f"Faces: {metrics.get('faces', 0)}")
            self.blinks_label.configure(text=f"Blinks: {metrics.get('blinks', 0)}")
        else:
            if not metrics:
                self.face_id_label.configure(text="Face ID: None")
                self.liveness_label.configure(text="Liveness: 0.0%")
                self.blinked_label.configure(text="Blinked: False")
                self.ear_label.configure(text="EAR: 0.00")
                self.mar_label.configure(text="MAR: 0.00")
                self.motion_label.configure(text="Motion: 0.00")
                self.nose_label.configure(text="Nose Length: 0.00")
                self.challenge_label.configure(text="Challenge: NONE")
                self.spoof_label.configure(text="Spoof: None")
            else:
                self.face_id_label.configure(text=f"Face ID: {metrics.get('face_id', 'None')}")
                self.liveness_label.configure(text=f"Liveness: {metrics.get('liveness', 0.0):.1f}%")
                self.blinked_label.configure(text=f"Blinked: {metrics.get('blinked', False)}")
                self.ear_label.configure(text=f"EAR: {metrics.get('ear', 0.0):.2f}")
                self.mar_label.configure(text=f"MAR: {metrics.get('mar', 0.0):.2f}")
                self.motion_label.configure(text=f"Motion: {metrics.get('motion', 0.0):.2f}")
                self.nose_label.configure(text=f"Nose Length: {metrics.get('nose_length', 0.0):.2f}")
                self.challenge_label.configure(text=f"Challenge: {metrics.get('challenge', 'NONE')}")
                self.spoof_label.configure(text=f"Spoof: {metrics.get('spoof_message', 'None')}")

    def start_processing(self):
        self.processing_thread = threading.Thread(target=self.face_recognizer.run,
                                                 args=(self.video_label, self.update_metrics))
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def quit(self):
        self.running = False
        self.face_recognizer.stop_thread = True
        self.root.quit()
        self.root.destroy()


def main():
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()