import cv2
import face_recognition
import pickle
import csv
import os
from datetime import datetime

# Attendance file
ATTENDANCE_FILE = "attendance.csv"

# Load known faces
try:
    with open("known_faces.pkl", "rb") as f:
        known_encodings, known_names = pickle.load(f)
except FileNotFoundError:
    known_encodings, known_names = [], []

print(f"[INFO] Loaded {len(known_names)} known faces.")
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])


def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            if f"{name},{date_str}" in f.read():
                return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
    print(f"[LOG] Marked attendance for {name}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

      
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        
        label_height = 35
        cv2.rectangle(frame, (left, bottom - label_height), (right, bottom), color, cv2.FILLED)

        
        font_scale = 1.0
        font_thickness = 2
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = left + (right - left - text_size[0]) // 2
        text_y = bottom - 10
        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        
        mark_attendance(name)

    cv2.imshow("Face Recognition & Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
