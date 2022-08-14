from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from typing import Tuple

path = pathlib.Path(__file__).parent


def target(X) -> np.ndarray:
    new_shape = [1]*len(X.shape[:-1]) + [X.shape[-1]]
    mu0, s0 = np.array([-0.5, -0.5]).reshape(new_shape), 0.3
    mu1, s1 = np.array([0.5, 0.5]).reshape(new_shape), 0.3
    return np.exp(-0.5*(np.sum(((X - mu0)/s0)**2, axis=-1))) - np.exp(-0.5*(np.sum(((X - mu1)/s1)**2, axis=-1)))


def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    Xobs = np.stack([np.random.uniform(-1, 1, 1000), np.random.uniform(-1, 1, 1000)], axis=-1)
    Yobs = target(Xobs)
    Yobs += np.random.normal(0, 0.1, size=Xobs.shape[0])
    return Xobs, Yobs


def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    Xa = np.random.multivariate_normal([-0.5, -0.5], [[0.3, 0.], [0., 0.3]], size=500)
    Xb = np.random.multivariate_normal([0.5, 0.5], [[0.3, 0.], [0., 0.3]], size=500)
    Xobs = np.concatenate([Xa, Xb], axis=0)
    Yobs = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    return Xobs, Yobs

# regression

f = plt.figure(figsize=[12, 5])
# surface
X = np.stack(np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101)), axis=-1)
Z = target(X)

ax = f.add_subplot(121, projection='3d', computed_zorder=False)
ax.plot_surface(X[..., 0], X[..., 1], Z, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title("target function")

# scatter
Xobs, Zobs = regression_data()
ax = f.add_subplot(122, projection="3d")
ax.scatter(Xobs[..., 0], Xobs[..., 1], Zobs, c=Zobs, marker=".", cmap="viridis")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title("target observations")

f.tight_layout()
f.savefig(path / "regression_target.png", transparent=True, dpi=300)


# classification
f, ax = plt.subplots(figsize=[5, 5])

Xobs, Yobs = classification_data()
ax.scatter(Xobs[..., 0], Xobs[..., 1], c=[mpl.cm.Set1.colors[int(i)] for i in Yobs], marker=".")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xticks([])
ax.set_yticks([])
f.tight_layout()
f.savefig(path / "classification_target.png", transparent=True, dpi=300)
