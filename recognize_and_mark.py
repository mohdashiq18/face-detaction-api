import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os

# ===== Load Known Encodings =====
with open("known_faces.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

print(f"[INFO] Loaded {len(known_names)} known faces.")

ATTENDANCE_FILE = "attendance.csv"

# Ensure attendance.csv exists with headers
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

def mark_attendance(name):
    now = datetime.now()
    dt = now.strftime('%Y-%m-%d')
    tm = now.strftime('%H:%M:%S')

    df = pd.read_csv(ATTENDANCE_FILE)
    if not ((df['Name'] == name) & (df['Date'] == dt)).any():
        with open(ATTENDANCE_FILE, "a") as f:
            f.write(f"{name},{dt},{tm}\n")
        print(f"[MARKED] {name} at {tm}")

def select_camera():
    print("[INFO] Searching for working cameras...")
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            continue
        print(f"[TEST] Camera index {index} — Press 'y' to select, 'n' to skip")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera Test", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('y'):
                cap.release()
                cv2.destroyAllWindows()
                print(f"[INFO] Selected camera index {index}")
                return index
            elif key == ord('n') or key == 27:
                break
        cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("No working camera selected.")

camera_index = select_camera()
cap = cv2.VideoCapture(camera_index)

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
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]

        mark_attendance(name)

        # Draw rectangle (green for known, red for unknown)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw label
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
