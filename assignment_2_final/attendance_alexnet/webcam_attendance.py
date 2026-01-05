import cv2
import numpy as np
import time
import tensorflow as tf
import os

# ================= CONFIG =================
MODEL_PATH = "model.h5"          # or model.keras if you switch later
DATASET_DIR = "dataset/train"    # used only to read class names
IMG_SIZE = 227

CONFIDENCE_THRESHOLD = 0.80      # minimum softmax confidence
PRESENCE_TIME = 3.0              # seconds required to confirm presence
# ==========================================


# ============ LOAD MODEL ==================
model = tf.keras.models.load_model(MODEL_PATH)

class_names = sorted(os.listdir(DATASET_DIR))
print("Loaded classes:", class_names)


# ========== FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ========== ATTENDANCE STATE ==============
attendance = set()
active_candidate = None
candidate_start_time = None

# ========== CAMERA ========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Attendance system started. Press 'q' to quit.")

# =========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)
        confidence = float(np.max(preds))
        class_id = int(np.argmax(preds))

        if confidence >= CONFIDENCE_THRESHOLD:
            student_name = class_names[class_id]

            if student_name == active_candidate:
                elapsed = current_time - candidate_start_time

                if elapsed >= PRESENCE_TIME:
                    if student_name not in attendance:
                        attendance.add(student_name)

                    label = f"{student_name} ✓ Present"
                    color = (0, 255, 0)
                else:
                    remaining = PRESENCE_TIME - elapsed
                    label = f"{student_name} ({remaining:.1f}s)"
                    color = (0, 255, 255)

            else:
                active_candidate = student_name
                candidate_start_time = current_time
                label = f"{student_name} (confirming...)"
                color = (255, 255, 0)

        else:
            active_candidate = None
            candidate_start_time = None
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # ===== SHOW ATTENDANCE COUNT =====
    cv2.putText(
        frame,
        f"Attendance Count: {len(attendance)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================================
cap.release()
cv2.destroyAllWindows()

print("\nFinal Attendance List:")
for student in sorted(attendance):
    print(student)
