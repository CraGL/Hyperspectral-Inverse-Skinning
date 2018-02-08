import autograd.numpy as np

from pymanopt.manifolds import Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions

# (1) Instantiate a manifold
manifold = Product( ( Euclidean(12), Grassmann(12, 3) ) )

# (2) Define the cost function (here using autograd.numpy)
def cost(X):
    p,B = X
    return np.linalg.norm( np.dot(B.T,p) )

problem = Problem(manifold=manifold, cost=cost)

# (3) Instantiate a Pymanopt solver
# solver = SteepestDescent()
solver = TrustRegions()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print(Xopt)
