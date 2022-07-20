import numpy as np
import matplotlib.pyplot as plt
import pathlib

path = pathlib.Path(__file__).parent


def trajectory(V0: float=4, T: float=10, dt: float=0.001, m: float=15, C: float=20):
    trajectory = []
    g = np.array([0., -9.81])
    X = np.zeros(2)
    V = np.array([-(V0/2)**0.5, (V0/2)**0.5])
    for t in np.arange(0, T, dt):
        if X[1] >= 0:
            A = g + C/m * V**2
            V += A*dt
            X += V*dt
        trajectory.append(X.tolist())
    return list(zip(*trajectory))


plt.style.use("bmh")
f, ax  = plt.subplots()

for v in [10, 8, 6, 4]:
    x, y = trajectory(V0=v)
    ax.plot(x, y, label=f"$V_0 = {v:d}$")
    ax.legend()
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("x")
ax.set_ylabel("y")
f.tight_layout()
f.savefig(path / "Trajectory.png", transparent=True)
plt.show()
