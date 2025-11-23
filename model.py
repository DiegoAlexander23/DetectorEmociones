import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_ds= tf.keras.preprocessing.image_dataset_from_directory(
    'archive(6)\\train', image_size=(48,48), color_mode='grayscale', batch_size=32
)
test_ds= tf.keras.preprocessing.image_dataset_from_directory(
    'archive(6)\\test', image_size=(48,48), color_mode='grayscale', batch_size=32
)
val_ds= tf.keras.preprocessing.image_dataset_from_directory(
    'archive(6)\\validation', image_size=(48,48), color_mode='grayscale', batch_size=32
)
class_names= train_ds.class_names
print(class_names)

train_ds= train_ds.map(lambda x,y: (x/255.0, y))
test_ds= test_ds.map(lambda x,y: (x/255.0, y))
val_ds= val_ds.map(lambda x,y: (x/255.0, y))


model= tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48,48,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history= model.fit(train_ds, epochs=15, validation_data= val_ds)

model.save('face_model.h5')