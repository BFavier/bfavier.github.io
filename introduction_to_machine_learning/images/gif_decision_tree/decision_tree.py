import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pathlib
import imageio
from sklearn import tree

path = pathlib.Path(__file__).parent

import sys
sys.path.append(str(path.parent))
from models_data import target, regression_data, classification_data

# regression

Xobs, Yobs = regression_data()
X = np.stack(np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)), axis=-1)

f = plt.figure(figsize=[10, 5])
ax = f.add_subplot(121, projection="3d")
ax2 = f.add_subplot(122)

ax.scatter(Xobs[:, 0], Xobs[:, 1], Yobs, marker=".", cmap="cividis", c=Yobs, depthshade=False, zorder=1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")

files = []
for i in range(1, 21):
    model = tree.DecisionTreeRegressor(max_leaf_nodes=i+1)
    model.fit(Xobs, Yobs)
    Y = model.predict(X.reshape(-1, 2)).reshape(X.shape[:2])
    tree.plot_tree(model, ax=ax2)

    ax.clear()
    ax.plot_surface(X[..., 0], X[..., 1], Y, rstride=1, cstride=1, cmap="viridis", zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")

    file_name = path / f"tree_reg{i}.png"
    files.append(file_name)
    f.savefig(file_name, transparent=True, dpi=300)
    print(file_name)

del test

# classification

Xobs, Yobs = classification_data()

f, ax = plt.subplots(figsize=[5, 5])
ax.scatter(Xobs[..., 0], Xobs[..., 1], c=[Yobs])
ax.set_xlim([0, len(X)])
ax.set_ylim([0, len(X)])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.yaxis.set_label_position("right")
ax.set_aspect("equal")
f.savefig(path / "tree.png")

files = []
for i in range(1, 21):
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

    # step
    file_name = path / f"tree{i}.png"
    files.append(file_name)
    f.savefig(file_name)
    print(file_name)
    # plt.show()
    plt.close(f)
    model_reg.split()
    model_class.split()

# IPython.embed()

with imageio.get_writer(path / "decision_tree.gif", mode='I', fps=2) as writer:
    for i in range(2):
        image = imageio.imread(path / f"tree.png")
        writer.append_data(image)
    for filename in files:
        image = imageio.imread(filename)
        writer.append_data(image)
    for i in range(2):
        writer.append_data(image)

