import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Type, Tuple, List
import os
import pathlib
import PIL

path = pathlib.Path(__file__).parent


class Visualizer(torch.nn.Module):

    def __init__(self, Optimizer: Type[torch.optim.Optimizer], lr: float = 1.0E-2, init: Tuple[float, float] = (-1., -1.)):
        super().__init__()
        self._xy = [init]
        self.params = torch.nn.parameter.Parameter(torch.tensor(self._xy[0], dtype=torch.float32, requires_grad=True))
        self.optimizer = Optimizer(self.parameters(), lr=lr)
    
    def step(self):
        self.optimizer.zero_grad()
        x, y = self.params
        loss = self.loss(x, y)
        loss.backward()
        self.optimizer.step()
        x, y = self.params.detach().tolist()
        self._xy.append((x, y))

    @staticmethod
    def loss(x, y):
        return 2*x**2-1.05*x**4 + x**6/6 + x*y +y**2
    
    @property
    def x(self) -> List[float]:
        return [x for x, _ in self._xy]
    
    @property
    def y(self) -> List[float]:
        return [y for _, y in self._xy]


x = np.linspace(-2, 2, 300)
y = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(x, y)
Z = Visualizer.loss(X, Y)
methods = ["gradient descent", "RMSprop", "Adadelta", "Adagrad", "Adam"]
visualizers = [Visualizer(opt) for opt in [torch.optim.SGD, torch.optim.RMSprop, torch.optim.Adadelta, torch.optim.Adagrad, torch.optim.Adam]]

f, ax = plt.subplots(figsize=[6, 4])
ax.set_xlabel("x")
ax.set_ylabel("y")
h = ax.imshow(Z, extent=(-2, 2, -2, 2), origin="lower", cmap="viridis")

colors = mpl.cm.Set1.colors
files = []

for step in range(1, 1001):
    ax.collections.clear()
    ax.lines.clear()
    for visualizer, color, name in zip(visualizers, colors, methods):
        visualizer.step()
        x, y = visualizer.x, visualizer.y
        if len(x) == 1:
            ax.scatter(x, y, color=color, marker=".", label=name)
        else:
            ax.plot(x, y, color=color, linewidth=1., label=name)
    f.legend()
    file_name = path / f"optimizer{step}.png"
    f.savefig(file_name, transparent=True, dpi=300)
    files.append(file_name)

image = PIL.Image.open(files[0])
images = [PIL.Image.open(file) for file in files[1:]]
image.save(path / 'optimizer_visualization.webp', save_all=True, append_images=images, loop=0, duration=10, disposal=2)
for file in files:
    os.remove(file)
