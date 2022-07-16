import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pathlib
import IPython

path = pathlib.Path(__file__).parent

# regression
N = 1000
Xobs = np.random.uniform(-1, 1, size=(N, 2))


def target(X) -> np.ndarray:
    x = X[..., None]
    mu0, s0 = np.array([[-0.5, -0.5]]), 0.3
    mu1, s1 = np.array([[0.5, 0.5]]), 0.3
    return np.exp(-0.5*(np.sum(((X - mu0)/s0)**2, axis=-1))) - np.exp(-0.5*(np.sum(((X - mu1)/s1)**2, axis=-1)))


def model(X: np.ndarray, Xobs: np.ndarray, Yobs: np.ndarray, k: int) -> np.ndarray:
    distances = np.sqrt(np.sum((X[..., None] - Xobs.T[None, ...])**2, axis=1))
    neighbours = np.argpartition(-distances, -k, axis=1)[:, -k:]
    return np.mean(Yobs[neighbours], axis=1)


Yobs = target(Xobs) + np.random.normal(0, 0.1, size=Xobs.shape[0])

X = np.stack(np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)), axis=-1)
Y = model(X.reshape(-1, 2), Xobs, Yobs, k=5).reshape(X.shape[:2])

f = plt.figure(figsize=[10, 5])
ax = f.add_subplot(121, projection="3d")
ax.scatter(Xobs[:, 0], Xobs[:, 1], Yobs, marker=".", color="k", depthshade=False, zorder=1)

ax.plot_surface(X[..., 0], X[..., 1], Y, cmap="cividis", zorder=0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")


# classification
ax = f.add_subplot(122)
Xa = np.random.multivariate_normal([-0.4, -0.4], [[0.2, 0.], [0., 0.2]], size=100)
Xb = np.random.multivariate_normal([0.4, 0.4], [[0.2, 0.], [0., 0.2]], size=100)
Xobs = np.concatenate([Xa, Xb], axis=0)
Yobs = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
Y = model(X.reshape(-1, 2), Xobs, Yobs, k=3).reshape(X.shape[:2])

R = Y < 0.5
G = Y >= 0.5
B = np.zeros(Y.shape)
image = np.stack([R, G, B], axis=-1)
image = (image * 55 + [[[200, 200, 200]]]).astype("uint8")

ax.imshow(image)
ax.scatter((Xa[:, 0]+1)/2*len(image), (Xa[:, 1]+1)/2*len(image), marker="o", color="r")
ax.scatter((Xb[:, 0]+1)/2*len(image), (Xb[:, 1]+1)/2*len(image), marker="v", color="g")
ax.set_xlim([0, image.shape[1]])
ax.set_ylim([0, image.shape[0]])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.yaxis.set_label_position("right")

f.savefig(path / "images" / "k_nearest.png")

plt.show()

IPython.embed()

