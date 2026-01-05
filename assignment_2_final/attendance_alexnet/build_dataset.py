import os
import cv2
import numpy as np
import random

# ---------------- CONFIG ----------------
RAW_DIR = "Dataset_LFPW"
OUT_DIR = "dataset"

IMG_SIZE = 227

TRAIN_COUNT = 40
VAL_COUNT = 10
TEST_COUNT = 10

random.seed(42)
# ----------------------------------------


def augment_image(img):
    h, w, _ = img.shape

    # rotation
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # brightness
    factor = random.uniform(0.8, 1.2)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    return img


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# create split folders
for split in ["train", "val", "test"]:
    ensure_dir(os.path.join(OUT_DIR, split))


for file in os.listdir(RAW_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    person_id = os.path.splitext(file)[0]  # person_001
    img_path = os.path.join(RAW_DIR, file)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    splits = (
        ("train", TRAIN_COUNT),
        ("val", VAL_COUNT),
        ("test", TEST_COUNT)
    )

    for split, count in splits:
        person_dir = os.path.join(OUT_DIR, split, person_id)
        ensure_dir(person_dir)

        for i in range(count):
            aug = augment_image(img)
            out_path = os.path.join(person_dir, f"{person_id}_{i}.jpg")
            cv2.imwrite(out_path, aug)

print("Dataset generation completed.")
