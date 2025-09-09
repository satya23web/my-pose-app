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

# --- MediaPipe and Accuracy Logic ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video_to_ragdoll(video_path: Path, ragdoll_path: Path):
    pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(str(video_path))
    all_frames_landmarks = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        frame_landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
        all_frames_landmarks.append(frame_landmarks)
    cap.release()
    with open(ragdoll_path, 'w') as f:
        json.dump(all_frames_landmarks, f)
    print(f"Ragdoll created at {ragdoll_path}")

def calculate_accuracy(live_landmarks, ref_landmarks):
    if not live_landmarks or not ref_landmarks: return 0.0
    vector_pairs = [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW), (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST), (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW), (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST), (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE), (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE), (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE), (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),]
    similarities = []
    for start_idx, end_idx in vector_pairs:
        live_vector = np.array([live_landmarks[end_idx]['x'] - live_landmarks[start_idx]['x'], live_landmarks[end_idx]['y'] - live_landmarks[start_idx]['y']])
        ref_vector = np.array([ref_landmarks[end_idx]['x'] - ref_landmarks[start_idx]['x'], ref_landmarks[end_idx]['y'] - ref_landmarks[start_idx]['y']])
        live_norm, ref_norm = np.linalg.norm(live_vector), np.linalg.norm(ref_vector)
        if live_norm > 0 and ref_norm > 0:
            similarity = np.dot(live_vector, ref_vector) / (live_norm * ref_norm)
            similarities.append(np.clip(similarity, -1.0, 1.0))
    return (np.mean(similarities) + 1) / 2 * 100 if similarities else 0.0

# --- FastAPI Endpoints ---

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = Path(file.filename).stem
    video_path = UPLOAD_DIR / file.filename
    ragdoll_path = RAGDOLL_DIR / f"{video_id}.json"
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    process_video_to_ragdoll(video_path, ragdoll_path)
    return JSONResponse({"status": "success", "video_id": video_id})

@app.get("/")
async def get_upload_page():
    with open("upload.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/practice/{video_id}")
async def get_practice_page(video_id: str):
    with open("practice.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await websocket.accept()
    ragdoll_path = RAGDOLL_DIR / f"{video_id}.json"
    if not ragdoll_path.exists():
        await websocket.close(code=1008, reason="Ragdoll not found")
        return
    with open(ragdoll_path, 'r') as f:
        all_poses = json.load(f)
    reference_poses = [pose for pose in all_poses if pose]
    frame_index = 0
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        data = await websocket.receive_text()
        
        # --- FIX: Check if the received data is not empty ---
        if data:
            try:
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

                # --- The rest of the processing logic is now safely inside the check ---
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_detector.process(frame_rgb)
                if frame_index >= len(reference_poses): frame_index = 0
                ref_pose_landmarks = reference_poses[frame_index]
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                    live_pose_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]
                    if ref_pose_landmarks:
                        h, w, _ = frame.shape
                        for connection in mp_pose.POSE_CONNECTIONS:
                            start_idx, end_idx = connection
                            start_point = (int(ref_pose_landmarks[start_idx]['x'] * w), int(ref_pose_landmarks[start_idx]['y'] * h))
                            end_point = (int(ref_pose_landmarks[end_idx]['x'] * w), int(ref_pose_landmarks[end_idx]['y'] * h))
                            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                    accuracy = calculate_accuracy(live_pose_landmarks, ref_pose_landmarks)
                    color = (0, 255, 0) if accuracy > 85 else (0, 0, 255)
                    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                frame_index += 1
                
                _, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_text("data:image/jpeg;base64," + base64.b64encode(buffer).decode())

            except Exception as e:
                print(f"Error processing frame: {e}")
                # This prevents a single bad frame from crashing the whole connection
                continue
