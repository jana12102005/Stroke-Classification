import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

data_dir = "A:/bloodclot/Processed_Data"
batch_size = 32
img_size = (224, 224)


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False 


x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


epochs = 50
model.fit(train_data, validation_data=val_data, epochs=epochs)


model.save("A:/bloodclot/stroke_model.h5")
