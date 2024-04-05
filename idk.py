import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Data paths (assuming your data structure is the same)
data_train_path = "Fruits_Vegetables/train"
data_test_path = "Fruits_Vegetables/test"
data_val_path = "Fruits_Vegetables/validation"

# Image dimensions (ensure these match VGG16 input size)
img_width = 224
img_height = 224

# Load and preprocess training data
data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)
data_train = data_train.map(lambda image, label: (tf.keras.applications.vgg16.preprocess_input(image), label))  # Preprocess for VGG16

# Load and preprocess validation data
data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    shuffle=False,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)
data_val = data_val.map(lambda image, label: (tf.keras.applications.vgg16.preprocess_input(image), label))  # Preprocess for VGG16

# Get class names
data_cat = data_train.class_names

# Visualize sample images
plt.figure(figsize=(10, 10))
for image, labels in data_train.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(data_cat[labels[i]])
        plt.axis('off')
plt.show()

# Define VGG16 model using transfer learning
base_model = tf.keras.applications.VGG16(
    weights='imagenet',  # Load pre-trained weights
    include_top=False,   # Exclude the top classifier layers
    input_shape=(img_width, img_height, 3)  # Specify input shape
)

# Freeze the base model layers (optional, can be fine-tuned later)
base_model.trainable = False  # Freeze base model layers

# Add new classification layers on top of the frozen VGG16
model = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),  # Dense layer with 1024 units and ReLU activation
    layers.Dropout(0.5),                    # Dropout for regularization
    layers.Dense(len(data_cat), activation='softmax')  # Output layer with softmax activation (one neuron per class)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
epochs_size = 25
history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)

# Plot training and validation accuracy/loss
epochs_range = range(epochs_size)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.show()

# Load and preprocess an image for prediction
image = "banana.jpg"
image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image)
img
