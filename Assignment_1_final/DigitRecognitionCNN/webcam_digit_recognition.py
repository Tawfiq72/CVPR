import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mnist_cnn_model.h5")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

roi_size = 300

while True:
    ret, frame = cap.read()
    if not ret:
        break



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    x1 = w//2 - roi_size//2
    y1 = h//2 - roi_size//2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    roi = gray[y1:y2, x1:x2]

    roi = cv2.GaussianBlur(roi, (5,5), 0)
    roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, _ = cv2.findContours(
    roi.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        roi = roi[y:y+h, x:x+w]

    roi = cv2.resize(roi, (28,28))
    roi = roi / 255.0
    roi = roi.reshape(1,28,28,1)

    pred = model.predict(roi, verbose=0)
    digit = np.argmax(pred)
    confidence = np.max(pred)

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(
        frame,
        f"{digit} ({confidence:.2f})",
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,0),
        2
    )

    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
