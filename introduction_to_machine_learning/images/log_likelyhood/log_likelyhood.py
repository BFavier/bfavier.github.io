import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import IPython
import pathlib

path = pathlib.Path(__file__).parent
mu, sigma = 1.52, 0.1
observations = np.random.normal(mu, sigma, (10000))
MU, SIGMA = np.meshgrid(np.linspace(mu-0.1*mu, mu+0.1*mu, 100), np.logspace(np.log10(sigma-0.5*sigma), np.log10(sigma+2*sigma), 100))

def normal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    """
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

def log_likelyhood(mu: np.ndarray, sigma: np.ndarray, observations: np.ndarray) -> float:
    """
    returns the log likelyhood of the observations given the parameters of the normal law
    """
    observations = np.expand_dims(observations, tuple(range(len(mu.shape))))
    mu, sigma = mu[..., None], sigma[..., None]
    return np.sum(-0.5*((observations - mu)/sigma)**2 - np.log(sigma*np.sqrt(2*np.pi)), axis=-1)

LL = log_likelyhood(MU, SIGMA, observations)

# LL -= LL.min() - 1.0E-9

# plot likelyhood
fig = plt.figure(figsize=[9, 4])
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(MU, SIGMA, LL, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("$\mu$")
ax.set_ylabel("$\sigma$")
ax.set_zlabel("log likelyhood")
ax.set_zticks([])

# plot histogram
ax = fig.add_subplot(122)
ax.hist(observations, bins=30, density=True)
x = np.linspace(observations.min(), observations.max(), 1000)
y = normal(x, mu, sigma)
ax.plot(x, y, linewidth=2.)
ax.set_yticks([])

fig.savefig(path / "log_likelyhood.png", transparent=True)
plt.show()

IPython.embed()