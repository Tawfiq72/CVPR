import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
IMG_SIZE = 227
BATCH_SIZE = 8   # keep small
EPOCHS = 20
# ----------------------------------------


# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU detected")
else:
    print("Running on CPU")


print("Loading datasets (streaming from disk)...")

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("Number of students:", NUM_CLASSES)


# Normalize on-the-fly
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


# ---------------- AlexNet ----------------
model = keras.Sequential([
    keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(96, (11,11), strides=4, activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),

    layers.Conv2D(256, (11,11), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),

    layers.Conv2D(384, (3,3), activation='relu'),
    layers.BatchNormalization(),

    layers.Conv2D(384, (3,3), activation='relu'),
    layers.BatchNormalization(),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),

    layers.Flatten(),

    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),

    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),

    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

model.save("model.h5")
print("Model saved as model.h5")
