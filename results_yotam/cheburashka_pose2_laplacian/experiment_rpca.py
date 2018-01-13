'''
Following http://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html
'''

from __future__ import print_function, division

from numpy import *
import sklearn.manifold

import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D

qs = loadtxt("local_subspace_recover.txt")
errors = loadtxt("local_subspace_recover_errors.txt")
ssv = loadtxt("local_subspace_recover_ssv.txt")

X = qs
X_r, err = sklearn.manifold.locally_linear_embedding( qs, n_neighbors=10, n_components=10 )

print("Done. Reconstruction error: %g" % err)

fig = plt.figure()

ax = fig.add_subplot(211, projection='3d')
ax.scatter(X[:, 0], X[:, 1], cmap=plt.cm.Spectral)

ax.set_title("Original data")
ax = fig.add_subplot(212)
ax.scatter(X_r[:, 0], X_r[:, 1], cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data')
plt.show()
