import math
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

    def __init__(self, Optimizer: Type[torch.optim.Optimizer], init: Tuple[float, float] = (-1., -1.), **kwargs):
        super().__init__()
        self._xy = [init]
        self.P = torch.nn.parameter.Parameter(torch.tensor(self._xy[0], dtype=torch.float32, requires_grad=True))
        self.optimizer = Optimizer(self.parameters(), **kwargs)
    
    def step(self):
        self.optimizer.zero_grad()
        x, y = self.P
        loss = self.loss(x, y)
        loss.backward()
        self.optimizer.step()
        x, y = self.P.detach().tolist()
        self._xy.append((x, y))

    @staticmethod
    def loss(x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        # return 2*x**2-1.05*x**4 + x**6/6 + x*y +y**2
        return torch.log(1 + (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2)
    
    @property
    def x(self) -> List[float]:
        return [x for x, _ in self._xy]
    
    @property
    def y(self) -> List[float]:
        return [y for _, y in self._xy]


x = np.linspace(-4.5, 4.5, 300)
y = np.linspace(-4.5, 4.5, 300)
X, Y = np.meshgrid(x, y)
Z = Visualizer.loss(X, Y)
methods = ["gradient descent", "Adagrad", "Adadelta", "RMSprop", "Adam"]
visualizers = [Visualizer(torch.optim.SGD, lr=1.0E-2), Visualizer(torch.optim.Adagrad, lr=0.8E-1), Visualizer(torch.optim.Adadelta, lr=1.0, rho=0.8), Visualizer(torch.optim.RMSprop, lr=0.8E-2), Visualizer(torch.optim.Adam, lr=1.0E-2)]

f, ax = plt.subplots(figsize=[5, 5], gridspec_kw={"left":0, "right": 1, "top": 1, "bottom": 0})
ax.imshow(Z, extent=(-4.5, 4.5, -4.5, 4.5), origin="lower", cmap="viridis")
ax.contour(Z, levels=15, colors="w", extent=(-4.5, 4.5, -4.5, 4.5), linewidths=0.1, origin="lower")

colors = mpl.cm.Set1.colors
scatters = []
files = []
F = 10

try:
    for step in range(0, 1001):
        # ax.collections.clear()  # correct me : removes contour
        for h in scatters:
            h.remove()
        scatters = []
        if step % F == 0:
            for visualizer, color, name in zip(visualizers, colors, methods):
                x, y = visualizer.x[-100:], visualizer.y[-100:]
                scatters.append(ax.scatter(x, y, color=color, marker=".", s=3., label=name, alpha=[math.exp(0.1*(i-len(x))) for i, _ in enumerate(x, start=1)]))
            leg = f.legend(markerscale=2)
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
            file_name = path / f"optimizer{step}.png"
            f.savefig(file_name, transparent=True, dpi=300)
            files.append(file_name)
        for visualizer in visualizers:
            visualizer.step()
except KeyboardInterrupt:
    pass

image = PIL.Image.open(files[0])
images = [PIL.Image.open(file) for file in files[1:]]
image.save(path / 'optimizer_visualization.webp', save_all=True, append_images=images, loop=0, duration=10, disposal=2)
for file in files:
    os.remove(file)
