import tensorflow as tf
import sys
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('../saved_model/mask_detector.h5')
img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)/255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
print(pred)
