import os
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from dataset import load_split
from utils import show_samples, save_evaluation

# Configure paths
BASE_DIR = r"C:\Users\Mohit Mehra\cat-and-dog"  # dataset location
IMG_SIZE = 64
USE_PCA = True
N_COMPONENTS = 200

def main():
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_split(BASE_DIR, img_size=IMG_SIZE)

    # Optional: visualize samples
    # show_samples(X_train, y_train, img_size=IMG_SIZE, n=6)

    steps = [("scaler", StandardScaler())]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=N_COMPONENTS, random_state=42)))
    steps.append(("svm", SVC(kernel="rbf", C=10, gamma="scale")))
    model = Pipeline(steps)

    print("Training SVM...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = save_evaluation(y_test, y_pred, out_dir="../results")
    print(f"Accuracy: {acc:.4f}")

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/svm_model.pkl")

if __name__ == "__main__":
    main()