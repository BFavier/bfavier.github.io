from telnetlib import IP
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import imageio
import pathlib
from itertools import chain
import IPython

plt.style.use("bmh")
path = pathlib.Path(__file__).parent


def target(X):
    return 3*X + 30 + np.random.normal(0, 10, size=X.shape)


def fit(Xobs: np.ndarray, Ytarget: np.ndarray, n: int) -> np.ndarray:
    """
    returns weights of polynom of degree n
    """
    Xobs = (Xobs - Xobs.mean())/Xobs.std()
    xobs = np.stack([Xobs**i for i in range(n, -1, -1)], axis=1)
    w, _, _, _ = np.linalg.lstsq(xobs, Ytarget, rcond=None)
    return w


def model(X: np.ndarray, w: np.ndarray, Xobs: np.ndarray) -> np.ndarray:
    """
    interpolates with given model
    """
    X = (X - Xobs.mean())/Xobs.std()
    n = len(w) - 1
    x = np.stack([X**i for i in range(n, -1, -1)], axis=1)
    return x@w


Xobs = np.linspace(7, 20, 31, dtype=np.longdouble)
X = np.linspace(7, 20, 1000, dtype=np.longdouble)
Ytarget = target(Xobs)
delta = Ytarget.max() - Ytarget.min()
ylims = [0, Ytarget.max() + 0.05*delta]

Xval = np.linspace(7, 20, 20, dtype=np.longdouble)
Yval = target(Xval)

weights = [fit(Xobs, Ytarget, n) for n in range(len(Xobs))]
MSE_train = [np.sum((Ytarget - model(Xobs, w, Xobs))**2) for w in weights]
MSE_val = [np.sum((Yval - model(Xval, w, Xobs))**2) for w in weights]

files = []
for n, w in enumerate(weights):
    f, axes = plt.subplots(figsize=[10, 5], ncols=2)
    axes[0].scatter(Xobs, Ytarget, label="données d'entraînement")
    axes[0].scatter(Xval, Yval, label="données de validation")
    axes[0].plot(X, model(X, w, Xobs), color="C3", label=f"polynôme de degrès {n}")
    axes[0].set_ylim(ylims)
    axes[0].set_xlabel("heure de publication")
    axes[0].set_ylabel("nombre de likes en 24h")
    axes[0].legend(loc="upper left")

    x_train = np.arange(n+1) - 0.25/2
    x_val = np.arange(n+1) + 0.25/2
    mse_train = MSE_train[:n+1]
    mse_val = MSE_val[:n+1]
    axes[1].bar(x_train, mse_train, width=0.25, color="C0")
    axes[1].bar(x_val, mse_val, width=0.25, color="C1")
    axes[1].set_ylabel("Somme des carrés des erreurs")
    axes[1].set_xlabel("degrés du polynôme")
    axes[1].yaxis.tick_right()
    axes[1].set_ylim([1.0E-1, max(max(MSE_train), max(MSE_val))])
    axes[1].set_yscale("log")
    axes[1].set_xlim([-1, len(weights)+1])
    xticks = range(0, len(weights)+1, 5)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([f"{tick}" for tick in xticks])
    axes[1].yaxis.set_label_position("right")

    file_name = path / f"poly{n}.png"
    files.append(file_name)
    f.savefig(file_name)
    print(file_name)
    plt.close(f)

with imageio.get_writer(path / "polynoms.gif", mode='I', fps=2) as writer:
    for filename in files:
        image = imageio.imread(filename)
        writer.append_data(image)
    for i in range(2):
        writer.append_data(image)
