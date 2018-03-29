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

def _is_seq( seq_or_one ):
    import autograd.builtins
    return type(seq_or_one) in (list,tuple,autograd.builtins.SequenceBox)

def _apply_to_one_or_more( seq_or_one, f ):
    if _is_seq( seq_or_one ):
        return [ f( x ) for x in seq_or_one ]
    else:
        return f( seq_or_one )

def compute_projection_mean( man, Ys ):
    """
    Given:
        man: A pymanopt.Manifold to optimize over
        Ys: A list of instances of objects in the manifold (or similar instances with the same first dimension).
    Returns:
        The member of `man` that is the projection mean of the Ys.
    """
    
    from pymanopt.tools.multi import multiprod, multitransp
    from pymanopt.manifolds import Product
    import autograd.numpy as np
    
    def objective( X ):
        e = 0.
        
        XXt = _apply_to_one_or_more( X, lambda M: np.dot( M, M.T ) )
        
        for Y in Ys:
            YYt = _apply_to_one_or_more( Y, lambda M: np.dot( M, M.T ) )
            
            if _is_seq( X ):
                diff = [ XXti - YYti for XXti, YYti in zip( XXt, YYt ) ]
                e += np.sum( [ np.sum(E*E) for E in diff ] )
            else:
                diff = XXt - YYt
                e += np.sum( diff*diff )
        
        return e
    
    def gradient( X ):
        grad = man.zerovec(X)
        
        XXt = _apply_to_one_or_more( X, lambda M: np.dot( M, M.T ) )
        
        for Y in Ys:
            YYt = _apply_to_one_or_more( Y, lambda M: np.dot( M, M.T ) )
            
            if _is_seq( X ):
                diff = [ XXti - YYti for XXti, YYti in zip( XXt, YYt ) ]
                grad += [ 4.*np.dot(diffi, Xi) for diffi, Xi in zip( diff, X ) ]
            else:
                diff = XXt - YYt
                grad += 4.*np.dot( diff, X )
        
        return grad
    
    solver = ConjugateGradient( maxiter=1000, mingradnorm=1e-10, minstepsize=1e-15 )
    # problem = Problem( man, cost=objective, grad=gradient, verbosity=2)
    problem = Problem( manifold = man, cost = objective, verbosity = 2 )
    return solver.solve(problem)
