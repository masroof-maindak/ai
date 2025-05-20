from functools import wraps
import json
import time

from preprocess import TARGET_H, TARGET_W
from split_data import TRAIN_LIST_PATH, TEST_LIST_PATH

import numpy as np
from PIL import Image


def load_images_and_labels(data_list: list[tuple[str, str]], list_name: str):
    num_samples: int = len(data_list)

    X = np.zeros((num_samples, TARGET_H, TARGET_W, 3), dtype=np.float32)
    y: list[str] = [None] * num_samples

    for i, (img_path, label) in enumerate(data_list):
        img = Image.open(img_path)
        X[i] = np.array(img) / 255.0  # Normalize image
        y[i] = label

    print(f"[ IMAGE LOADING ] Succesfully fetched `{list_name}`.")
    return X, y


def load_dataset():
    with open(TRAIN_LIST_PATH) as f:
        train_list = json.load(f)
    with open(TEST_LIST_PATH) as f:
        test_list = json.load(f)

    X_train, y_train = load_images_and_labels(train_list, "train list")
    X_test, y_test = load_images_and_labels(test_list, "test list")

    return X_train, y_train, X_test, y_test


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {end_time - start_time:.6f} seconds to execute."
        )
        return result

    return wrapper
