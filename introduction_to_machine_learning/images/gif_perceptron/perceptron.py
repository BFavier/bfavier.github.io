import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import imageio
import pathlib
import IPython

path = pathlib.Path(__file__).parent


X1, X2 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

files = []
for i, theta in enumerate(np.linspace(0, 2*np.pi, 100)):
    a, b = np.cos(theta), np.sin(theta)
    Y = np.tanh(3*(a*X1+b*X2))
    f = plt.figure(figsize=[5, 5])
    ax = f.add_subplot(projection="3d")
    ax.plot_surface(X1, X2, Y, cmap="plasma", cstride=1, rstride=1)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    file_name = path / f"preceptron{i}.png"
    files.append(file_name)
    f.tight_layout()
    f.savefig(file_name)
    plt.close(f)
    print(file_name)


with imageio.get_writer(path / "perceptron.gif", mode='I', fps=24) as writer:
    for filename in files:
        image = imageio.imread(filename)
        writer.append_data(image)