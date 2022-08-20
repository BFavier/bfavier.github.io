import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import PIL
import os
from sklearn import ensemble, tree
from sklearn.datasets import load_boston

path = pathlib.Path(__file__).parent

import sys
sys.path.append(str(path.parent))
from models_data import target, regression_data, classification_data


# regression

boston = load_boston()
X = boston.data
y = boston.target
model = tree.DecisionTreeRegressor(max_leaf_nodes=10, max_depth=100)
model.fit(X, y)
y_pred = model.predict(X)
f, ax = plt.subplots(figsize=[5, 5])
