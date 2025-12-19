import tensorflow as tf
import os

MODEL_PATH = os.path.join('..', 'saved_model', 'mask_detector.h5')
SAVE_TFLITE = os.path.join('..', 'saved_model', 'mask_detector.tflite')

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(SAVE_TFLITE, 'wb') as f:
    f.write(tflite_model)
print('Saved', SAVE_TFLITE)
