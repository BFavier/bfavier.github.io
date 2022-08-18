import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pathlib
import IPython
import matplotlib as mpl
import torch


path = pathlib.Path(__file__).parent

import sys
sys.path.append(str(path.parent))
from models_data import target, regression_data, classification_data


def create_points(i_layer: int, n_neurons: int, R: float = 1.) -> list[tuple[float, float]]:
    x = 5*R*i_layer
    L = 2*n_neurons*R + 0.5*R*(n_neurons - 1)
    ys = np.linspace(-L/2, L/2, n_neurons) if n_neurons > 1 else [0]
    return [(x, y) for y in ys]


# explaination
colors = mpl.cm.Set1.colors
f, ax = plt.subplots(figsize=[5, 5])
R = 1
layers = [create_points(i, n, R) for i, n in enumerate([2, 5, 5, 1])]
layer_colors = [colors[1], colors[0], colors[0], colors[2]]
hard_connections = [2, 4, 1]
for points, color in zip(layers, layer_colors):
    for point in points:
        ax.add_patch(mpl.patches.Circle(point, R, color=color, zorder=1, antialiased=True))
for i_layer, hard_connection in enumerate(hard_connections):
    for i, point_B in enumerate(layers[i_layer+1], start=1):
        for point_A in layers[i_layer]:
            xs = [p[0] for p in (point_A, point_B)]
            ys = [p[1] for p in (point_A, point_B)]
            plt.plot(xs, ys, color="k", linewidth=1 if i == hard_connection else 0.2, antialiased=True, zorder=0)
xs = [point[0] for points in layers for point in points]
ys = [point[1] for points in layers for point in points]
ax.set_xlim([min(xs)-2*R, max(xs)+2*R])
ax.set_ylim([min(ys)-2*R, max(ys)+2*R])
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
f.tight_layout()

f.savefig(path / "neural_network_explaination.png", transparent=True, dpi=300)

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
ax.set_title("k nearest neighbours (k=5)")

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
ax.set_title("k nearest neighbours (k=5)")

f.savefig(path / "k_nearest_classification.png", transparent=True, dpi=300)

