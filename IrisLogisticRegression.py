import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_mldata
from IPython.display import display, HTML
pd.set_option("display.max_rows",50)

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


zmin = Z.min()
zmax=Z.max()
levels=np.linspace(zmin, zmax, 1000)
print(zmin,zmax)

fig = plt.figure()
ax = fig.add_subplot(111)

# Put the result into a color plot
Z = Z.reshape(xx.shape)

#plt.pcolormesh(xx, yy, Z, vmin=Z.min(), vmax=Z.max(), cmap=plt.cm.Paired)

plt.contourf(xx, yy, Z, levels, vmin=zmin, vmax=zmax, cmap=plt.cm.Paired,alpha=0.7)
plt.contour(xx, yy, Z, levels, vmin=zmin, vmax=zmax, colors='black',alpha=0.7)


# Plot also the training points
plt.scatter(X[:, 0], X[:, 1],c=Y, cmap=plt.cm.Paired, s=40)
plt.xlabel('Sepal length', size=20)
plt.ylabel('Sepal width', size=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
print(iris['target_names'])

ax.text(4.5,4.5,'Iris setosa',size=20)
ax.text(5.5,1.75,'Iris versicolor',size=20)
ax.text(7.5,4.5,'Iris virginica',size=20)

plt.show()

fig.savefig('IrisClassification.pdf', format='pdf', dpi=900)


plt.show()