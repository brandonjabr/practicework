import warnings
warnings.filterwarnings("ignore")

%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from datascience import *
from pandas import Series, DataFrame
import pandas as pd
import sklearn as sk
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_mldata
from IPython.display import display, HTML
pd.set_option("display.max_rows",50)
matplotlib.rcParams['figure.figsize'] = 12, 8

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

fig = plt.figure()
ax = fig.add_subplot(111)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired,shading='gourand',alpha=0.9)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1],c=Y, cmap=plt.cm.Paired, s=80)
plt.xlabel('Sepal length', size=20)
plt.ylabel('Sepal width', size=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
print(iris['target_names'])

ax.annotate('Iris Setosa', xy=(4.5, 4.5), xytext=(4.5, 4.5))

plt.show()