from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import cv2
import numpy as np
import pickle
import csv
import os
from datetime import datetime
import base64
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ATTENDANCE_FILE = "attendance.csv"
KNOWN_FACES_FILE = "known_faces.pkl"
FACE_RECOGNITION_TOLERANCE = 0.5

# Load known faces encodings and names
try:
    with open(KNOWN_FACES_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)
except FileNotFoundError:
    known_encodings, known_names = [], []

# Create attendance file if not exists with headers
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time"])

def mark_attendance(name: str):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Read attendance file once to check existing records
    with open(ATTENDANCE_FILE, "r") as f:
        records = f.readlines()
    # Check if attendance already marked today for the name
    if any(f"{name},{date_str}" in record for record in records):
        return

    # Append new attendance record
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, date_str, time_str])
    print(f"[LOG] Attendance marked for {name} at {time_str}")

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        np_img = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detections = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            mark_attendance(name)

        detections.append({
            "name": name,
            "box": [top, right, bottom, left]
        })

        # Draw rectangle and label on the frame for visualization
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Encode annotated image to base64 to send back as a preview
    _, buffer = cv2.imencode(".jpg", frame)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={"detections": detections, "image": encoded_img})
