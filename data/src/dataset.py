import os
import cv2
import numpy as np
from tqdm import tqdm

def load_split(base_dir, img_size=64):
    """
    Loads training_set and test_set from base_dir.
    Returns: X_train, y_train, X_test, y_test
    Labels: 0 = cat, 1 = dog
    """
    def load_folder(folder):
        X, y = [], []
        for label, sub in enumerate(["cats", "dogs"]):
            path = os.path.join(folder, sub)
            files = os.listdir(path)
            for f in tqdm(files, desc=f"Loading {sub} from {os.path.basename(folder)}"):
                fp = os.path.join(path, f)
                img = cv2.imread(fp)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                X.append(img.flatten())  # flatten to 1D vector
                y.append(label)
        return np.array(X), np.array(y)

    train_dir = os.path.join(base_dir, "training_set")
    test_dir  = os.path.join(base_dir, "test_set")

    X_train, y_train = load_folder(train_dir)
    X_test, y_test   = load_folder(test_dir)
    return X_train, y_train, X_test, y_test