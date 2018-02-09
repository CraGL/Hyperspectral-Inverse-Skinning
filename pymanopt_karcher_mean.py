from __future__ import print_function, division

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient

## Modified code from nelder_mead.py.

def compute_centroid(man, x):
    """
    Compute the centroid as Karcher mean of points x belonging to the manifold
    man.
    """
    n = len(x)

    def objective(y):  # weighted Frechet variance
        acc = 0
        for i in range(n):
            acc += man.dist(y, x[i]) ** 2
        return acc / 2

    def gradient(y):
        g = man.zerovec(y)
        for i in range(n):
            g -= man.log(y, x[i])
        return g

    # TODO: manopt runs a few TR iterations here. For us to do this, we either
    #       need to work out the Hessian of the Frechet variance by hand or
    #       implement approximations for the Hessian to use in the TR solver.
    #       This is because we cannot implement the Frechet variance with
    #       theano and compute the Hessian automatically due to dependency on
    #       the manifold-dependent distance function.
    # solver = SteepestDescent(maxiter=150)
    solver = ConjugateGradient(maxiter=1000,mingradnorm=1e-10,minstepsize=1e-15)
    problem = Problem(man, cost=objective, grad=gradient, verbosity=2)
    return solver.solve(problem)
