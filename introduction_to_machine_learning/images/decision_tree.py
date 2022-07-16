import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pathlib
import imageio
import IPython

path = pathlib.Path(__file__).parent


class Branche:

    def __hash__(self) -> int:
        return hash(self.index, self.threshold, self.purity_gain)

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            observations, array of shape (N, D)

        Y : np.ndarray
            target, array of shape (N,)
        """
        index, threshold, purity_gain = max(self._all_splits(X, Y), key=lambda x: x[2])
        self.index = index
        self.threshold = threshold
        self.purity_gain = purity_gain
        mask_left = self._mask_left(X)
        self.left = (X[mask_left], Y[mask_left])
        self.right = (X[~mask_left], Y[~mask_left])

    def _all_splits(self, X: np.ndarray, Y: np.ndarray) -> list:
        """
        returns the list of (index, threshold, purity_gain)
        for each possible splits
        """
        splits = []
        N, D = X.shape
        for index in range(D):
            x = X[:, index]
            values = np.sort(np.unique(x))
            thresholds = 0.5*values[1:] + 0.5*values[:-1]
            for threshold in thresholds:
                mask_left = (x <= threshold)
                f_left = mask_left.sum() / len(Y)
                f_right = 1 - f_left
                purity_gain = Y.var() - f_left*Y[mask_left].var() - f_right*Y[~mask_left].var()
                splits.append((index, threshold, purity_gain))
        return splits
    
    def _mask_left(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.index] <= self.threshold

    def leafs(self):
        """
        returns an iterable of (parent, left, leaf) for each leaf of the tree with :
            parent : the parent Branche object 
            left : boolean which is true if the leaf is the left branche of the parent
            leaf : tuple of (X, Y) of values contained in the leaf
        """
        leafs = (self.left, self.right)
        is_left = (True, False)
        for leaf, left in zip(leafs, is_left):
            if isinstance(leaf, Branche):
                for l in leaf.leafs():
                    yield l
            else:
                if len(leaf[0]) > 1:
                    yield self, left, leaf

    def propagate(self, X: np.ndarray, super_mask: np.ndarray = True):
        """
        returns an iterable of (mask, value) for each leaf of the tree
        """
        leafs = (self.left, self.right)
        left_mask = self._mask_left(X)
        masks = (super_mask & left_mask, super_mask & ~left_mask)
        for leaf, mask in zip(leafs, masks):
            if isinstance(leaf, Branche):
                for sub_mask, value in leaf.propagate(X, mask):
                    yield sub_mask, value
            else:
                yield  mask, leaf[1].mean()


class DecisionTree:

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.stump = Branche(X, Y)
    
    def __call__(self, X: np.ndarray):
        """
        predict with the decision tree on the given observations

        Parameters
        ----------
        X : np.ndarray
            array of shape (N, D)
        """
        Y = np.zeros(len(X))
        for mask, value in self.stump.propagate(X):
            Y[mask] = value
        return Y
    
    def split(self):
        """
        preforms the single best posible split of a leaf
        """
        all_splits = [(parent, left, Branche(*leaf)) for parent, left, leaf in self.stump.leafs()]
        parent, left, branche = max(all_splits, key=lambda x: x[2].purity_gain)
        if left:
            parent.left = branche
        else:
            parent.right = branche

N = 1000
Xobs = np.random.uniform(-1, 1, size=(N, 2))


def target(X) -> np.ndarray:
    x = X[..., None]
    mu0, s0 = np.array([[-0.5, -0.5]]), 0.3
    mu1, s1 = np.array([[0.5, 0.5]]), 0.3
    return np.exp(-0.5*(np.sum(((X - mu0)/s0)**2, axis=-1))) - np.exp(-0.5*(np.sum(((X - mu1)/s1)**2, axis=-1)))


Yobs = target(Xobs) + np.random.normal(0, 0.1, size=Xobs.shape[0])
X = np.stack(np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)), axis=-1)
model_reg = DecisionTree(Xobs, Yobs)

Xa = np.random.multivariate_normal([-0.4, -0.4], [[0.2, 0.], [0., 0.2]], size=100)
Xb = np.random.multivariate_normal([0.4, 0.4], [[0.2, 0.], [0., 0.2]], size=100)
Xobs_scatter = np.concatenate([Xa, Xb], axis=0)
Yobs_scatter = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
model_class = DecisionTree(Xobs_scatter, Yobs_scatter)

f = plt.figure(figsize=[10, 5])
ax = f.add_subplot(121, projection="3d")
ax.scatter(Xobs[:, 0], Xobs[:, 1], Yobs, marker=".", cmap="cividis", c=Yobs, depthshade=False, zorder=1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
ax = f.add_subplot(122)
ax.scatter((Xa[:, 0]+1)/2*len(X), (Xa[:, 1]+1)/2*len(X), marker="o", color="r")
ax.scatter((Xb[:, 0]+1)/2*len(X), (Xb[:, 1]+1)/2*len(X), marker="v", color="g")
ax.set_xlim([0, len(X)])
ax.set_ylim([0, len(X)])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.yaxis.set_label_position("right")
ax.set_aspect("equal")
f.savefig(path / "images" / "gif_decision_tree" / "tree.png")

files = []
for i in range(20):
    f = plt.figure(figsize=[10, 5])

    # regression
    Y = model_reg(X.reshape(-1, 2)).reshape(X.shape[:2])
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
    Yimg = model_class(X.reshape(-1, 2)).reshape(X.shape[:2])

    R = Yimg < 0.5
    G = Yimg >= 0.5
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

    # step
    file_name = path / "images" / "gif_decision_tree" / f"tree{i}.png"
    files.append(file_name)
    f.savefig(file_name)
    print(file_name)
    # plt.show()
    plt.close(f)
    model_reg.split()
    model_class.split()

# IPython.embed()

with imageio.get_writer(path / "images" / "decision_tree.gif", mode='I', fps=2) as writer:
    for i in range(2):
        image = imageio.imread(path / "images" / "gif_decision_tree" / f"tree.png")
        writer.append_data(image)
    for filename in files:
        image = imageio.imread(filename)
        writer.append_data(image)
    for i in range(2):
        writer.append_data(image)

