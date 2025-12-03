import os
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataset import load_split
from utils import show_samples

# Configure paths
BASE_DIR = r"C:\Users\Mohit Mehra\PycharmProjects\SCT_ML_3\data\cat-and-dog"
IMG_SIZE = 64
USE_PCA = True      # set False to train directly on raw pixels
N_COMPONENTS = 200  # tune based on speed/accuracy tradeoff

def main():
    # Load data
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_split(BASE_DIR, img_size=IMG_SIZE)

    # Optional: visualize a few samples
    # show_samples(X_train, y_train, img_size=IMG_SIZE, n=6)

    # Build pipeline: scale -> (optional PCA) -> SVM
    steps = [("scaler", StandardScaler())]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=N_COMPONENTS, random_state=42)))
    steps.append(("svm", SVC(kernel="rbf", C=10, gamma="scale")))  # try kernel='linear' too
    model = Pipeline(steps)

    print("Training SVM...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Optional: quick visualization of a few predictions
    n_show = 6
    idxs = np.random.choice(len(X_test), size=min(n_show, len(X_test)), replace=False)
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(idxs, 1):
        img = X_test[idx].reshape(IMG_SIZE, IMG_SIZE, 3)
        plt.subplot(1, n_show, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Pred: " + ("Cat" if y_pred[idx] == 0 else "Dog"))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()