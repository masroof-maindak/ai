import os
import json
import random
from collections import defaultdict

# Directories
PROCESSED_DIR = "processed_images"
TRAIN_LIST_PATH = "train_data.json"
TEST_LIST_PATH = "test_data.json"

SPLIT_RATIO = 0.8  # 80% training, 20% testing


def extract_class_from_filename(filename):
    return filename.split("_")[0]


def split_dataset():
    if os.path.exists(TRAIN_LIST_PATH) and os.path.exists(TEST_LIST_PATH):
        print("[ SPLITTING ] Data found. Skipping.")
        return

    random.seed(42)  # Reproducibility
    class_to_files = defaultdict(list)

    # Collect all image paths grouped by class
    for fname in os.listdir(PROCESSED_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        label = extract_class_from_filename(fname)
        full_path = os.path.join(PROCESSED_DIR, fname)
        class_to_files[label].append(full_path)

    train_data = []
    test_data = []

    for label, files in class_to_files.items():
        random.shuffle(files)
        split_idx = int(len(files) * SPLIT_RATIO)
        train = [(f, label) for f in files[:split_idx]]
        test = [(f, label) for f in files[split_idx:]]

        train_data.extend(train)
        test_data.extend(test)

    # Save to JSON
    with open(TRAIN_LIST_PATH, "w") as f:
        json.dump(train_data, f)

    with open(TEST_LIST_PATH, "w") as f:
        json.dump(test_data, f)

    print(
        f"[ SPLITTING ] Saved {len(train_data)} training and {len(test_data)} testing examples."
    )


if __name__ == "__main__":
    split_dataset()
