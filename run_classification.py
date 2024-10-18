import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime

train_dir = "./Birds_25/train"
val_dir = "./Birds_25/valid"

image_size = (100, 100)  # Increase image size for better feature extraction
batch_size = 64


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

inputs = keras.Input(shape=(100, 100, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(32, (3, 3), activation="relu")(x)
x = layers.SpatialDropout2D(0.2)(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.SpatialDropout2D(0.2)(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(128, (3, 3), activation="relu")(x)
x = layers.SpatialDropout2D(0.3)(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(256, (3, 3), activation="relu")(x)
x = layers.SpatialDropout2D(0.3)(x)
x = layers.GlobalMaxPooling2D()(x)
outputs = layers.Dense(25, activation="softmax")(x)
model_1 = keras.Model(inputs, outputs)

model_1.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=keras.optimizers.Adam(learning_rate=0.001)  # Adjust learning rate
)

logdir = "./logs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

model_1.fit(
    train_ds,
    epochs=50,  # Increase the number of epochs if needed
    validation_data=val_ds,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

model_1.evaluate(train_ds)
