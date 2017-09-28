from numpy import *

def min_quad_with_linear_constraints( P, q, A, b ):
    '''
    Given:
        P, q: Dense arrays representing an energy `.5 x^T P x + q^T x`
        A, b: Dense arrays for linear constraints `A x = b` to satisfy.
    Returns:
        The minimum of the energy subject to the constraints.
    '''
    
    P = asfarray( P )
    q = asfarray( q )
    A = asfarray( A )
    b = asfarray( b )
    
    ## The quadratic part of the energy should be square.
    assert P.shape[0] == P.shape[1]
    n = P.shape[1]
    ## The matrices must have the same number of degrees-of-freedom.
    assert A.shape[1] == n
    assert q.shape[0] == n
    ## There should be a constraint RHS for each constraint.
    assert A.shape[0] == b.shape[0]
    
    # method = 'lagrange'
    method = 'lsq'
    
    if method == 'lagrange':
        ## Let's set up a Lagrange multiplier system.
        system = zeros( ( n + A.shape[0], n + A.shape[0] ) )
        ## The upper left is P
        system[:n,:n] = P
        ## The lower left is A
        system[n:,:n] = A
        ## The upper right is A.T
        system[:n,n:] = A.T
        
        ## The right hand side is -q above b
        rhs = zeros( system.shape[1] )
        rhs[:n] = -q
        rhs[n:] = b
    elif method == 'lsq':
        system = array(P)
        w = 1e9
        system += w*A.T.dot(A)
        rhs = -q + w*A.T.dot(b)
    else:
        raise NotImplementedError
    
    return linalg.solve( system, rhs )

def binary_search( x0, direction, max_step, G, h, threshold = 1e-10, epsilon = 0.0 ):
    '''
    Given:
        x0: An initial known valid state.
        direction: A search direction.
        max_step: The maximum scalar multiple of `direction` to consider. Must be positive.
        G, h: Linear inequality constraints `G x <= h`, where `x = x0 + step*direction` (this matches cvxopt.lp and qp).
        threshold (optional, default 1e-10): The threshold below which to stop the binary search for `step`.
        epsilon (optional, default 0): An epsilon value to add to `h`, so that `G x <= h + epsilon`.
    Returns `x0 + step*direction`, where `step` is as large as possible while `G ( x0 + step*direction ) <= h`.
    '''
    
    assert max_step > 0
    assert threshold > 0
    
    x0 = asfarray( x0 )
    direction = asfarray( direction )
    
    ## If we have a non-zero epsilon, just add it to h.
    if epsilon != 0: h = asfarray( h ) + epsilon
    
    ## The initial guess must be valid.
    assert ( G.dot( x0 ) <= h ).all()
    
    ## Start with as big a step as possible.
    good = 0.
    bad = max_step
    assert bad >= good
    
    step = max_step
    while bad - good > threshold:
        if ( G.dot( x0 + step*direction ) <= h ).all():
            good = step
        else:
            bad = step
        step = .5*( good + bad )
    
    print( "binary_search final step:", step )
    
    return x0 + step*direction
