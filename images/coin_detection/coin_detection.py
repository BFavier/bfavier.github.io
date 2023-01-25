from skimage import data
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import pathlib

path = pathlib.Path(__file__).parent

coins = data.coins()
tr = coins > 150
edges = canny(coins)
fill_coins = ndi.binary_fill_holes(edges)
coins_cleaned = morphology.remove_small_objects(fill_coins, 21)
labeled_coins, _ = ndi.label(coins_cleaned)
image_label_overlay = label2rgb(labeled_coins, image=coins, bg_label=0)

f, axes = plt.subplots(figsize=[4*3, 4*2], ncols=3, nrows=2)
axes[0][0].imshow(coins, cmap="Greys")
axes[0][0].set_title("original image")
axes[0][1].imshow(tr, cmap="Greys_r")
axes[0][1].set_title("thresholded")
axes[0][2].imshow(edges, cmap="Greys_r")
axes[0][2].set_title("edge detection")
axes[1][0].imshow(fill_coins, cmap="Greys_r")
axes[1][0].set_title("holes filled")
axes[1][1].imshow(coins_cleaned, cmap="Greys_r")
axes[1][1].set_title("removed small objects")
axes[1][2].imshow(image_label_overlay)
axes[1][2].set_title("detected objects")
for c in axes:
    for ax in c:
        ax.axis('off')
f.tight_layout()
f.savefig(path / "coins.png", dpi=300, transparent=True)
plt.show()