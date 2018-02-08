from __future__ import division, print_function

from numpy import *

def principal_angles( A, B ):
    '''
    Given:
        A: An orthonormal matrix whose columns are directions in the ambient space
        B: An orthonormal matrix whose columns are directions in the ambient space
    Returns:
        The 1D array of principal angles between A and B
    '''
    return linalg.svd( A.T.dot( B ), compute_uv = False )

def canonical_point( p, B ):
    '''
    Given:
        p: a point in R^n on the flat.
        B: An orthonormal matrix whose columns are directions in the ambient space R^n.
    Returns:
        The point on the flat closest to the origin.
    '''
    
    ## E = | p + B z |^2 = ( p + B z ) : ( p + B z ) = M : M
    ## min_z E
    ## dE = 2M : ( B dz ) = 2 B' M : dz
    ## dE/dz = 0 = B' M = B' p + B' B z <=> z = (B'B)^(-1) (-B'p)
    ## The point is: p - B (B'B)^(-1) B' p
    
    p_closest = p - B.dot( linalg.solve( B.T.dot( B ), B.T.dot(p) ) )
    # p_closest = p - B.dot( linalg.pinv( B.T.dot( B ) ).dot( B.T.dot(p) ) )
    return p_closest
