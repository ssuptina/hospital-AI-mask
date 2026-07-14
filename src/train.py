import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import build_model
import os

train_dir = '../dataset/'
import os
print("Dataset path:", os.path.abspath(train_dir))
print("Folders:", os.listdir(train_dir))

batch_size = 32

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255,
                             rotation_range=20, zoom_range=0.2,
                             horizontal_flip=True, brightness_range=[0.8,1.2])

train_gen = datagen.flow_from_directory(train_dir, target_size=(224,224),
                                        subset='training', batch_size=batch_size)
val_gen = datagen.flow_from_directory(train_dir, target_size=(224,224),
                                      subset='validation', batch_size=batch_size)

model = build_model(num_classes=len(train_gen.class_indices))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=5)

os.makedirs('../saved_model', exist_ok=True)
model.save('../saved_model/mask_detector.h5')
