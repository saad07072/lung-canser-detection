# create_dataset.py
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

OUT_DIR = "mini_dataset"
CLASSES = ["normal", "cancer"]
SAMPLES_PER_CLASS = 100
SIZE = (128,128)

os.makedirs(OUT_DIR, exist_ok=True)
for c in CLASSES:
    d = os.path.join(OUT_DIR, c)
    os.makedirs(d, exist_ok=True)

def make_normal(size):
    # smooth lung-like blobs
    img = Image.new("L", size, color=30)
    draw = ImageDraw.Draw(img)
    for _ in range(3):
        x0 = random.randint(5, size[0]//2)
        y0 = random.randint(10, size[1]//2)
        x1 = random.randint(size[0]//2, size[0]-5)
        y1 = random.randint(size[1]//2, size[1]-10)
        draw.ellipse([x0,y0,x1,y1], fill=random.randint(40,80))
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    # add low random noise
    arr = np.array(img).astype(np.uint8)
    arr = np.clip(arr + np.random.normal(0,4,arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def make_cancer(size):
    img = make_normal(size)
    draw = ImageDraw.Draw(img)
    # add small bright nodules
    for _ in range(random.randint(1,4)):
        cx = random.randint(20, size[0]-20)
        cy = random.randint(20, size[1]-20)
        r = random.randint(4,12)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=random.randint(170,255))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(img)
    arr = np.clip(arr + np.random.normal(0,6,arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

for i in range(SAMPLES_PER_CLASS):
    normal = make_normal(SIZE)
    normal.save(os.path.join(OUT_DIR,"normal", f"normal_{i:03d}.png"))
    cancer = make_cancer(SIZE)
    cancer.save(os.path.join(OUT_DIR,"cancer", f"cancer_{i:03d}.png"))

print("Mini dataset created in", OUT_DIR)
