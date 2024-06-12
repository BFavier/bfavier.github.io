import numpy as np
import matplotlib.pyplot as plt
import pathlib
import PIL
import os

path = pathlib.Path(__file__).parent


def trajectory(V0: float=100, T: float=1.0, dt: float=0.001, m: float=15, C: float=50):
    trajectory = []
    g = np.array([0., -9.81])
    X = np.zeros(2)
    V = np.array([-(V0/2)**0.5, (V0/2)**0.5])
    for i, t in enumerate(np.arange(0, T, dt)):
        if X[1] >= 0:
            A = g + C/m * np.linalg.norm(V) * -V
            V += A*dt
            X += V*dt
        else:
            A = 0
            V = 0
            X = X
        if i % 10 == 0:
            trajectory.append(X.tolist())
    return list(zip(*trajectory))


plt.style.use("bmh")
f, ax  = plt.subplots()
values = [200, 100, 50, 30]
trajectories = [trajectory(V0=v) for v in values]
experimental = trajectory(V0=89)
N = len(trajectories[0][0])
files = []

for i in range(N):
    ax.clear()
    x, y = experimental
    ax.scatter(x, y, marker="x", s=0.6, color="k")
    for (x, y), v in zip(trajectories, values):
        ax.plot(x[:i+1], y[:i+1], label=f"$V_0 = {v:d}$")
    ax.legend()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([-0.8, 0.])
    ax.set_ylim([0., 0.6])
    ax.set_aspect("equal")
    ax.set_title("Trajectory for various initial velocity")
    # f.tight_layout()
    file_name = path / f"trajectory{i}.png"
    print(file_name)
    f.savefig(file_name, transparent=True, dpi=300)
    files.append(file_name)

# with imageio.get_writer(path / "trajectory.gif", mode='I', fps=24) as writer:
#     for filename in files:
#         image = imageio.imread(filename)
#         writer.append_data(image)

image = PIL.Image.open(files[0]).convert('P')
images = [PIL.Image.open(file).convert('P') for file in files[1:]]
image.save(path / "trajectory.gif", save_all=True, append_images=images, loop=0, duration=3, transparency=0)
for file in files:
    os.remove(file)
