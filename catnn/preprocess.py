import os

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Paths
INPUT_DIR = "images"
OUTPUT_DIR = "processed_images"
MARKER_FILE = os.path.join(OUTPUT_DIR, ".preprocessing_done")

# Parameters
TARGET_H: int = 160
TARGET_W: int = 160
SALT_PEPPER_PROB: float = 0.05


def resize_image(img):
    return img.resize((TARGET_H, TARGET_W), Image.BILINEAR)


def to_grayscale(img):
    gray = ImageOps.grayscale(img)
    return Image.merge("RGB", (gray, gray, gray))


def apply_gaussian_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def apply_salt_and_pepper(img, prob):
    arr = np.array(img)
    noisy = arr.copy()
    rnd = np.random.rand(*arr.shape[:2])

    noisy[rnd < (prob / 2)] = 255  # Salt
    noisy[(rnd >= (prob / 2)) & (rnd < prob)] = 0  # Pepper

    return Image.fromarray(noisy.astype(np.uint8))


def preprocess_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(MARKER_FILE):
        print("[ PREPROCESSING ] Data found. Skipping.")
        return

    print("[ PREPROCESSING ] Initiating...")

    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(INPUT_DIR, filename)
        img = Image.open(img_path).convert("RGB")
        img = resize_image(img)

        base_name = os.path.splitext(filename)[0]

        # Save resized
        img.save(os.path.join(OUTPUT_DIR, f"{base_name}_resized.jpg"))

        # Grayscale
        gray = to_grayscale(img)
        gray.save(os.path.join(OUTPUT_DIR, f"{base_name}_gray.jpg"))

        # Gaussian Blur
        blur = apply_gaussian_blur(img)
        blur.save(os.path.join(OUTPUT_DIR, f"{base_name}_blur.jpg"))

        # Salt and Pepper
        snp = apply_salt_and_pepper(img, SALT_PEPPER_PROB)
        snp.save(os.path.join(OUTPUT_DIR, f"{base_name}_snp.jpg"))

    # Create marker file to prevent rerunning
    with open(MARKER_FILE, "w") as f:
        _ = f.write("done")

    print("[ PREPROCESSING ] Completed successfully.")


if __name__ == "__main__":
    preprocess_images()
