"""
Real-Time Mask Detection via Webcam or Video File
Usage:
  python detect_mask_video.py              # webcam
  python detect_mask_video.py --video path/to/video.mp4
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import DETECTION_CONFIG, CLASS_LABELS, CLASS_COLORS

# ── Load models ───────────────────────────────────────────────────────────────
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("Loading face detector...")
face_net = cv2.dnn.readNetFromCaffe(
    DETECTION_CONFIG["FACE_PROTOTXT"],
    DETECTION_CONFIG["FACE_MODEL"]
)

print("Loading mask classifier...")
mask_model = load_model(DETECTION_CONFIG["MASK_MODEL"])
print("Models loaded!")

def detect_and_predict(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs, preds = [], [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > DETECTION_CONFIG["FACE_CONFIDENCE"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX   = min(w - 1, endX)
            endY   = min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.resize(face, (224, 224))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if faces:
        batch = np.vstack(faces)
        preds = mask_model.predict(batch, verbose=0)

    return locs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Path to video file (default: webcam)")
    args = ap.parse_args()

    src = args.video if args.video else 0
    cap = cv2.VideoCapture(src)

    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        locs, preds = detect_and_predict(frame)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            class_idx = np.argmax(pred)
            label = CLASS_LABELS[class_idx]
            color = CLASS_COLORS[class_idx]
            confidence = pred[class_idx] * 100

            label_text = f"{label}: {confidence:.1f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label_text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Mask Detection - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
