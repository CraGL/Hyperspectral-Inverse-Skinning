from __future__ import print_function, division

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import pdb

points_2d = np.array([[1.,0.], [0.,1.],[1.,2.],[2.,1.],[1.,1.]])

hull = sp.spatial.qhull.Delaunay(X).convex_hull

print(hull)
