import face_recognition
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

KNOWN_FACES_FILE = "known_faces.pkl"
known_encodings = []
known_names = []

for filename in os.listdir('students/'):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join('students', filename)
        image = face_recognition.load_image_file(path)
        # print(image.shape)
        encoding = face_recognition.face_encodings(image)[0]
        print(encoding)
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(filename)[0])
        print(f"[INFO] Encoded {filename}")

# Save encodings
with open(KNOWN_FACES_FILE, 'wb') as f:
    pickle.dump((known_encodings, known_names), f)

print(f"[DONE] Encoded {len(known_names)} faces.")
