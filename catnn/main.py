import json

from preprocess import preprocess_images, TARGET_H, TARGET_W
from split_data import split_dataset, TRAIN_LIST_PATH, TEST_LIST_PATH

import numpy as np
from PIL import Image


# CHECK: I probably won't need this but let's keep it around for now
def batch_flatten_images(data_list: list[tuple[str, str]]):
    """
    Batch load and flatten images to numpy arrays.

    Args:
        data_list (list of tuples): [(image_path, class), ...]

    Returns:
        X: np.ndarray of shape (num_samples, Height*Width*3)
        y: list of labels
    """

    num_samples = len(data_list)
    feature_dim = TARGET_H * TARGET_W * 3
    loop_modulus: int = int(num_samples / 6)

    X = np.zeros((num_samples, feature_dim), dtype=np.float32)
    y: list[str] = []

    for i, (img_path, label) in enumerate(data_list):
        img = Image.open(img_path)
        arr = np.array(img)
        X[i, :] = arr.flatten() / 255.0  # Normalize pixels
        y.append(label)

        if i % loop_modulus == 0:
            print(f"[ FLATTENING ] Processed {i}/{num_samples} images")

    print(f"[ FLATTENING ] Completed successfully.")
    return X, y


def main():
    preprocess_images()
    split_dataset()

    with open(TRAIN_LIST_PATH) as f:
        train_list = json.load(f)
    with open(TEST_LIST_PATH) as f:
        test_list = json.load(f)

    X_train, y_train = batch_flatten_images(train_list)
    X_test, y_test = batch_flatten_images(test_list)


if __name__ == "__main__":
    main()
