import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from dataset import load_split
from utils import save_evaluation

# Configure paths
BASE_DIR = r"C:\Users\Mohit Mehra\cat-and-dog"  # dataset location
IMG_SIZE = 64
USE_PCA = True
N_COMPONENTS = 200

def main():
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_split(BASE_DIR, img_size=IMG_SIZE)

    # Build pipeline: scale -> (optional PCA) -> SVM
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

    # Save trained model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/svm_model.pkl")

    # --- NEW: Display predictions grid ---
    n_show = 12  # number of images to display
    idxs = np.random.choice(len(X_test), size=min(n_show, len(X_test)), replace=False)
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(idxs, 1):
        img = X_test[idx].reshape(IMG_SIZE, IMG_SIZE, 3)  # reshape back to image
        plt.subplot(2, n_show // 2, i)  # 2 rows
        plt.imshow(img)
        plt.axis('off')
        plt.title("Pred: " + ("Cat" if y_pred[idx] == 0 else "Dog"))
    plt.tight_layout()
    plt.savefig("../results/predictions_grid.png")  # save for LinkedIn screenshot
    plt.show()

if __name__ == "__main__":
    main()