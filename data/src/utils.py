import numpy as np
import matplotlib.pyplot as plt

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