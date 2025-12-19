import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('../saved_model/mask_detector.h5')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (224,224))/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = np.argmax(pred)
    cv2.putText(frame, f"Class: {label}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
