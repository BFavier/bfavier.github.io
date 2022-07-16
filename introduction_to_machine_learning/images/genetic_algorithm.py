import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import imageio
import pathlib
import IPython

path = pathlib.Path(__file__).parent

Xgrid = np.stack(np.meshgrid(np.linspace(
    0, 24, 100), np.linspace(0, 100, 100)))


def target(X: np.ndarray) -> np.ndarray:
    return 200 - 0.1*(X[0] - 12)**2 - 0.01*(X[1] - 50)**2 - 0.2*X[1]


Xobs = np.stack([np.random.uniform(0, 24, 1000),
                np.random.uniform(0, 100, 1000)])
Ytarget = target(Xobs) + np.random.uniform(-10, 10, Xobs.shape[1])

P0 = np.random.normal(0, 0.3, size=(1, 5))


def normalize(X: np.ndarray):
    """
    normalize the given inputs

    Parameters
    ----------
    X : np.ndarray
        inputs of shape (2, *)

    Returns
    -------
    np.ndarray :
        normalized inputs of shape (2, *)
    """
    mean = np.expand_dims(Xobs.mean(axis=1), tuple(range(1, len(X.shape))))
    std = np.expand_dims(Xobs.std(axis=1), tuple(range(1, len(X.shape))))
    return (X - mean)/(std + 1.0E-9)


def rescale(ypred: np.ndarray):
    """
    rescale the given target

    Parameters
    ----------
    ypred : np.ndarray
        normalized predictions, array of floats of shape (*)

    Returns
    -------
    np.ndarray :
        Ypred, array of floats of shape (*)
    """
    return ypred * (Ytarget.std() + 1.0E-9) + Ytarget.mean()


def model(Xobs: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Return the evaluation of each model for each observation

    Parameters
    ----------
    Xobs : np.ndarray
        observations, array of floats of shape (2, *n_obs)
    P : np.ndarray
        model parameters, array of floats of shape (n_models, 5)

    Returns
    -------
    np.ndarray
        Ypred, array of floats of shape (n_models, *n_obs)
    """
    xobs = normalize(Xobs)
    X1, X2 = xobs
    n_dims = len(X1.shape)
    A, B, C, D, E = P.transpose()
    X1, X2 = X1[None, ...], X2[None, ...]
    A, B, C, D, E = [np.expand_dims(
        v, tuple(len(v.shape) + i for i in range(n_dims))) for v in (A, B, C, D, E)]
    return rescale(A*X1**2 + B*X2**2 + C*X1 + D*X2 + E)


def loss(Ypred: np.ndarray, Ytarget: np.ndarray) -> np.ndarray:
    """
    Returns the loss of each model prediction

    Parameters
    ----------
    Ypred : np.ndarray
        predictions of each model on each observation, array of floats of shape (n_models, *n_obs)
    Ytarget : np.ndarray
        target of each observation, array of shape (*n_obs)

    Returns
    -------
    np.ndarray :
        losses of each model, array of floats of shape (n_models)
    """
    return np.sum(((Ypred - Ytarget[None, :])**2).reshape(Ypred.shape[0], -1), axis=-1)


def create_axes():
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(projection='3d')
    return fig, ax


def draw_observations(ax):
    ax.scatter(Xobs[0], Xobs[1], Ytarget, marker=".", color="k")
    ax.set_xlabel("$X_1$", fontsize=10)
    ax.set_ylabel("$X_2$", fontsize=10)
    ax.set_zlabel("Y", fontsize=10)


def next_P(P0: np.ndarray, n_childs: int = 100) -> np.ndarray:
    """
    generate 'n_childs' for each model parameters, and pick the 'len(P0)' best models

    Parameters
    ----------
    P0 : np.ndarray
        parameters of each model, of shape (n_models, 5)
    
    Returns
    -------
    np.ndarray
        P, new set of best parameters
    """
    n_models = P0.shape[0]
    delta = np.random.normal(0., 0.01, size=(n_childs,)+tuple(P0.shape))
    new_P = (P0[None, ...]+delta).reshape(-1, 5)
    P = np.concatenate([P0, new_P])
    losses = loss(model(Xobs, P), Ytarget)
    sort = sorted([(l, i) for i, l in enumerate(losses)], key=lambda x: x[0])
    _, indexes = zip(*sort[:n_models])
    return P[indexes, ...]

f, ax = create_axes()
draw_observations(ax)
ax.set_title(f"step 0:")
ax.set_zlim([Ytarget.min(), Ytarget.max()])
f.tight_layout()
f.savefig(path / "images" / "gif_genetic" / f"genetic0.png")
plt.close(f)

files = []
for step in range(100):
    file_name = path / "images" / "gif_genetic" / f"genetic{step+1}.png"
    f, ax = create_axes()
    draw_observations(ax)
    Ypred = model(Xgrid, P0)
    l = loss(model(Xobs, P0), Ytarget)[0]
    ax.plot_surface(Xgrid[0], Xgrid[1], Ypred[0],
                    rstride=1, cstride=1, cmap="inferno", vmin=Ytarget.min(), vmax=Ytarget.max())
    params = ", ".join(f"{p:.3g}" for p in P0[0])
    ax.set_title(f"step {step+1}: loss={l:.3g}, (a, b, c, d, e, f)=({params})")
    ax.set_zlim([Ytarget.min(), Ytarget.max()])
    f.tight_layout()
    f.savefig(file_name)
    print(file_name)
    files.append(file_name)
    plt.close(f)
    P0 = next_P(P0)
# IPython.embed()


with imageio.get_writer(path / "images" / "genetic.gif", mode='I', fps=10) as writer:
    base = imageio.imread(path / "images" / "gif_genetic" / f"genetic0.png")
    for i in range(10):
        writer.append_data(base)
    for filename in files:
        image = imageio.imread(filename)
        writer.append_data(image)
