import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pathlib
import IPython
import matplotlib as mpl

path = pathlib.Path(__file__).parent

import sys
sys.path.append(str(path.parent))
from models_data import target, regression_data, classification_data


def model(X: np.ndarray, Xobs: np.ndarray, Yobs: np.ndarray, k: int) -> np.ndarray:
    distances = np.sqrt(np.sum((X[..., None] - Xobs.T[None, ...])**2, axis=1))
    neighbours = np.argpartition(-distances, -k, axis=1)[:, -k:]
    return np.mean(Yobs[neighbours], axis=1)


# regression

Xobs, Yobs = regression_data()

X = np.stack(np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)), axis=-1)
Y = model(X.reshape(-1, 2), Xobs, Yobs, k=5).reshape(X.shape[:2])

f = plt.figure(figsize=[5, 5])
ax = f.add_subplot(111, projection="3d")

ax.plot_surface(X[..., 0], X[..., 1], Y, rstride=1, cstride=1, cmap="viridis", zorder=0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")

f.savefig(path / "k_nearest_regression.png", transparent=True, dpi=300)

# classification

f, ax = plt.subplots(figsize=[5, 5])
Xobs, Yobs = classification_data()
is_b = Yobs.astype(bool)
Xa, Xb = Xobs[~is_b], Xobs[is_b]
X = np.stack(np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-2, 2, 500)), axis=-1)
Y = model(X.reshape(-1, 2), Xobs, Yobs, k=3).reshape(X.shape[:2])

R = Y < 0.5
B = Y >= 0.5
G = np.zeros(Y.shape)
image = np.stack([R, G, B], axis=-1)
image = (image * 55 + [[[200, 200, 200]]]).astype("uint8")

ax.imshow(image, extent=(-2, 2, -2, 2), origin="lower")
ax.scatter(Xobs[..., 0], Xobs[..., 1], c=[mpl.cm.Set1.colors[int(i)] for i in Yobs], marker=".")
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.yaxis.set_label_position("right")

f.savefig(path / "k_nearest_classification.png", transparent=True, dpi=300)

