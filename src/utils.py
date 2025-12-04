import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def show_samples(X, y, img_size=64, n=6):
    idxs = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(idxs, 1):
        img = X[idx].reshape(img_size, img_size, 3)
        plt.subplot(1, n, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Cat" if y[idx] == 0 else "Dog")
    plt.tight_layout()
    plt.show()

def save_evaluation(y_true, y_pred, out_dir="../results"):
    os.makedirs(out_dir, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    with open(os.path.join(out_dir, "accuracy.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    report = classification_report(y_true, y_pred, target_names=["Cat", "Dog"])
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    return acc