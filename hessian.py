from numpy import *
import scipy.optimize

def hessian( x0, f = None, grad = None, epsilon = 1e-7 ):
    '''
    Given an n-dimensional point x0,
    a scalar function `f` that can be evaluated at x0 OR its gradient function `grad`,
    and an optional epsilon parameter,
    returns the n-by-n finite difference hessian of f.
    '''
    
    assert f is None or grad is None
    
    if grad is None:
        grad = lambda x: scipy.optimize.approx_fprime( x, f, epsilon )
    
    g0 = grad( x0 )
    n = len(x0)
    H = identity( n )
    for i in range( n ):
        xoff = x0.copy()
        xoff[i] += epsilon
        H[i] = ( grad( xoff ) - g0 )/epsilon
    
    return H
