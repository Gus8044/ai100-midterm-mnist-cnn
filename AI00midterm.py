import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ===============================
# 1. Load MNIST Dataset
# ===============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ===============================
# 2. Preprocess Data
# ===============================

# Normalize pixel values (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to include channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ===============================
# 3. Build CNN Model
# ===============================

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# ===============================
# 4. Compile Model
# ===============================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 5. Train Model
# ===============================

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# ===============================
# 6. Evaluate Model
# ===============================

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_accuracy)