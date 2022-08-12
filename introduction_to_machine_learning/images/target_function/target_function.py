from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pathlib

path = pathlib.Path(__file__).parent


def target(X) -> np.ndarray:
    new_shape = [1]*len(X.shape[:-1]) + [X.shape[-1]]
    mu0, s0 = np.array([-0.5, -0.5]).reshape(new_shape), 0.3
    mu1, s1 = np.array([0.5, 0.5]).reshape(new_shape), 0.3
    return np.exp(-0.5*(np.sum(((X - mu0)/s0)**2, axis=-1))) - np.exp(-0.5*(np.sum(((X - mu1)/s1)**2, axis=-1)))


f = plt.figure(figsize=[10, 5])
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
Xobs = np.stack([np.random.uniform(-1, 1, 1000), np.random.uniform(-1, 1, 1000)], axis=-1)
Zobs = target(Xobs)
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
f.savefig(path / "target_function.png", transparent=True, dpi=300)
