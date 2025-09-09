# main.py
import cv2
import mediapipe as mp
import json
import numpy as np
import base64
import os
import shutil
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

# --- Basic Setup ---
app = FastAPI()
UPLOAD_DIR = Path("uploads")
RAGDOLL_DIR = Path("ragdolls")
UPLOAD_DIR.mkdir(exist_ok=True)
RAGDOLL_DIR.mkdir(exist_ok=True)

# --- MediaPipe and Helper Logic ---
mp_pose = mp.solutions.pose

# Define connections for each body part
L_ARM = {mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.LEFT_THUMB}
R_ARM = {mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY, mp_pose.PoseLandmark.RIGHT_INDEX, mp_pose.PoseLandmark.RIGHT_THUMB}
L_LEG = {mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX}
R_LEG = {mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX}
TORSO = {mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP}
HEAD = {mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT}

# Define colors for each body part (in BGR format)
BODY_PART_COLORS = {
    "L_ARM": (0, 255, 255),  # Yellow
    "R_ARM": (0, 165, 255),  # Orange
    "L_LEG": (255, 255, 0),  # Cyan
    "R_LEG": (255, 0, 255),  # Magenta
    "TORSO": (230, 230, 230),# White
    "HEAD": (255, 192, 203) # Pink
}

# Custom drawing function to color-code the skeleton
def draw_colored_landmarks(image, landmarks, is_live_data=True):
    h, w, _ = image.shape
    landmark_points = []
    if is_live_data:
        if landmarks:
            for lm in landmarks.landmark:
                landmark_points.append((int(lm.x * w), int(lm.y * h)))
    else:
        for lm in landmarks:
            landmark_points.append((int(lm['x'] * w), int(lm['y'] * h)))
    if not landmark_points: return

    for part_name, color in BODY_PART_COLORS.items():
        part_landmarks = globals()[part_name]
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in part_landmarks and end_idx in part_landmarks:
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    start_point, end_point = landmark_points[start_idx], landmark_points[end_idx]
                    cv2.line(image, start_point, end_point, color, 2, cv2.LINE_AA)
    for point in landmark_points:
        cv2.circle(image, point, 3, (0, 0, 255), -1, cv2.LINE_AA)

def smooth_pose_data(pose_data, window_size=5):
    if not pose_data: return []
    smoothed_data, num_landmarks = [], len(pose_data[0]) if pose_data[0] else 0
    if num_landmarks == 0: return pose_data
    for i in range(len(pose_data)):
        current_frame_landmarks = []
        if not pose_data[i]:
            smoothed_data.append([])
            continue
        for landmark_idx in range(num_landmarks):
            start = max(0, i - window_size + 1)
            window = pose_data[start:i+1]
            x_coords = [frame[landmark_idx]['x'] for frame in window if frame and len(frame) > landmark_idx]
            y_coords = [frame[landmark_idx]['y'] for frame in window if frame and len(frame) > landmark_idx]
            z_coords = [frame[landmark_idx]['z'] for frame in window if frame and len(frame) > landmark_idx]
            avg_x, avg_y, avg_z = (sum(x_coords) / len(x_coords) if x_coords else 0, sum(y_coords) / len(y_coords) if y_coords else 0, sum(z_coords) / len(z_coords) if z_coords else 0)
            current_frame_landmarks.append({'x': avg_x, 'y': avg_y, 'z': avg_z})
        smoothed_data.append(current_frame_landmarks)
    return smoothed_data

def process_video_to_ragdoll(video_path: Path, ragdoll_path: Path):
    pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(str(video_path))
    all_frames_landmarks = []
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector
