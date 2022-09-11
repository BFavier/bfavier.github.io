from skimage.filters import gabor_kernel
from itertools import count
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import PIL
import numpy as np
import os
import pathlib


path = pathlib.Path(__file__).parent
image = resize(imread(path / "face.png"), (26, 26), order=0).mean(axis=-1)/255
filter = gabor_kernel(1/4, theta=0.25*np.pi, bandwidth=1, n_stds=3).astype(float)
filter /= np.abs(filter).max()


def applied(image: np.ndarray, filter: np.ndarray, x_offset: int, y_offset: int) -> tuple[np.ndarray, float]:
    mult = 0.
    img = image.copy()
    h, w = image.shape
    dh, dw = filter.shape
    for i in range(0, dh):
        for j in range(0, dw):
            y, x = y_offset + i - dh//2, x_offset + j - dw//2
            if 0 <= y < h and 0 <= x < w and 0 <= i < dh and 0 <= j < dw:
                mult += image[y, x] * filter[i, j]
                img[y, x] = (filter[i, j] + 1.) * 0.5
    return img, mult


f, axes = plt.subplots(figsize=[8, 4], ncols=2)
counter = count()
files = []
h, w = image.shape
result = np.full((h, w, 4), [int(round(0.5 * 255))] * 3 + [0], dtype="uint8")
for i in range(h):
    for j in range(w):
        ax1, ax2 = axes
        new_im, res = applied(image, filter, j, i)
        result[i, j, :] = [int(round((res+1) / 2 * 255))]*3 + [255]
        ax1.clear()
        ax2.clear()
        ax1.axis("off")
        ax2.axis("off")
        ax1.imshow(new_im, cmap="gray", vmin=0, vmax=1)
        ax2.imshow(result)
        file = f"image{next(counter)}.png"
        f.savefig(path / file, transparent=True, dpi=300)
        print(file)
        files.append(path / file)


image = PIL.Image.open(files[0]).convert('P')
images = [PIL.Image.open(file).convert('P') for file in files[1:]]
image.save(path / 'feature_extraction.webp', save_all=True, append_images=images, loop=0, duration=10, disposal=2)

for file in files:
    os.remove(file)

if __name__ == "__main__":
    import IPython
    IPython.embed()