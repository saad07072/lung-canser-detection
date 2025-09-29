# train_model.py
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------
# Config
# -----------------------
DATA_DIR = "mini_dataset"
IMG_SIZE = (128, 128)
BATCH = 16
EPOCHS = 6
MODEL_OUT = "mini_model.h5"
SAMPLES_PER_CLASS = 100  # small demo dataset

# -----------------------
# Synthetic dataset generator (runs only if DATA_DIR missing)
# -----------------------
def make_normal(size):
    img = Image.new("L", size, color=30)
    draw = ImageDraw.Draw(img)
    for _ in range(3):
        x0 = random.randint(5, size[0]//2)
        y0 = random.randint(10, size[1]//2)
        x1 = random.randint(size[0]//2, size[0]-5)
        y1 = random.randint(size[1]//2, size[1]-10)
        draw.ellipse([x0,y0,x1,y1], fill=random.randint(40,80))
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    arr = np.array(img).astype(np.uint8)
    arr = np.clip(arr + np.random.normal(0,4,arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def make_cancer(size):
    img = make_normal(size)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1,4)):
        cx = random.randint(20, size[0]-20)
        cy = random.randint(20, size[1]-20)
        r = random.randint(4,12)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=random.randint(170,255))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(img)
    arr = np.clip(arr + np.random.normal(0,6,arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def create_mini_dataset(out_dir=DATA_DIR, samples_per_class=SAMPLES_PER_CLASS, size=IMG_SIZE):
    print("Creating synthetic demo dataset at", out_dir)
    classes = ["normal", "cancer"]
    os.makedirs(out_dir, exist_ok=True)
    for c in classes:
        d = os.path.join(out_dir, c)
        os.makedirs(d, exist_ok=True)

    for i in range(samples_per_class):
        normal = make_normal(size)
        normal.save(os.path.join(out_dir, "normal", f"normal_{i:03d}.png"))
        cancer = make_cancer(size)
        cancer.save(os.path.join(out_dir, "cancer", f"cancer_{i:03d}.png"))
    print("Synthetic dataset created: {} images per class.".format(samples_per_class))

# -----------------------
# Ensure dataset exists
# -----------------------
if not os.path.isdir(DATA_DIR):
    create_mini_dataset()

# quick check: show counts
def count_images(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                total += 1
    return total

print("Total images found in", DATA_DIR, ":", count_images(DATA_DIR))

# -----------------------
# Load datasets
# -----------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH
)

# -----------------------
# Model
# -----------------------
normalization_layer = layers.Rescaling(1./255)

model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (1,)),
    normalization_layer,
    layers.Conv2D(16,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy','AUC'])

model.summary()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save(MODEL_OUT)
print("Saved model to", MODEL_OUT)
