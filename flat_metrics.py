from __future__ import division, print_function

import numpy
from numpy import *

def principal_angles( A, B, orthonormal = False ):
    '''
    Given:
        A: A matrix whose columns are directions in the ambient space
        B: A matrix whose columns are directions in the ambient space
    Returns:
        The 1D array of cosines of principal angles between A and B
    '''
    ## We can skip orthonormalization if the matrices are known to be orthonormal.
    if not orthonormal:
        A = orthonormalize( A )
        B = orthonormalize( B )
    return linalg.svd( A.T.dot( B ), compute_uv = False )

def orthonormalize( A, threshold = None ):
    '''
    Given:
        A: A matrix whose columns are directions in the ambient space
        threshold (optional): The threshold cutoff for the rank of A.
    Returns:
        A with orthonormal columns (e.g. A.T*A = I). Note that the returned matrix
        may have fewer columns than the input.
    '''
    
    if threshold is None: threshold = 1e-9
    
    A = asarray( A )
    
    U, s, V = numpy.linalg.svd( A.T, full_matrices = False, compute_uv = True )
    ## The first index less than threshold
    stop_s = len(s) - numpy.searchsorted( s[::-1], threshold )
    ## Take the first stop_s rows of V as the columns of the result.
    
    return V[:stop_s].T

def canonical_point( p, B ):
    '''
    Given:
        p: a point in R^n on the flat.
        B: An orthonormal matrix whose columns are directions in the ambient space R^n.
    Returns:
        The point on the flat closest to the origin.
    '''
    
    p = asarray( p )
    B = asarray( B )
    
    ## E = | p + B z |^2 = ( p + B z ) : ( p + B z ) = M : M
    ## min_z E
    ## dE = 2M : ( B dz ) = 2 B' M : dz
    ## dE/dz = 0 = B' M = B' p + B' B z <=> z = (B'B)^(-1) (-B'p)
    ## The point is: p - B (B'B)^(-1) B' p
    
    p_closest = p - B.dot( linalg.solve( B.T.dot( B ), B.T.dot(p) ) )
    # p_closest = p - B.dot( linalg.pinv( B.T.dot( B ) ).dot( B.T.dot(p) ) )
    return p_closest

def test_principal_angles():
    A = array([[1,0,0]]).T
    B = array([[0,1,0]]).T
    angles = principal_angles( A, B, orthonormal = True )
    numpy.testing.assert_allclose( angles, zeros(1) )
    ## Check if orthonormalization is working.
    angles = principal_angles( 2*A, -10*B )
    numpy.testing.assert_allclose( angles, zeros(1) )
    
    A = array([[1,0,0],[0,0,1]]).T
    B = array([[0,1,0]]).T
    angles = principal_angles( A, B )
    numpy.testing.assert_allclose( angles, zeros(1) )
    
    A = array([[1,1,0]]).T
    B = array([[1,0,0]]).T
    angles = principal_angles( A, B )
    numpy.testing.assert_allclose( angles, (1/sqrt(2))*ones(1) )
    
    A = array([[1,0,0],[0,0,1]]).T
    B = array([[1,1,0]]).T
    angles = principal_angles( A, B )
    numpy.testing.assert_allclose( angles, cos(pi/4)*ones(1) )

def test_orthonormalize():
    A = identity(3)
    ortho = orthonormalize( A )
    numpy.testing.assert_allclose( ortho, identity(3) )
    
    A = hstack( [ identity(3), identity(3) ] )
    ortho = orthonormalize( A )
    ## They should be the same now up to sign changes and permutations.
    ## Let's assume no permutations and flip signs to that the largest element
    ## in each column is positive.
    ortho2 = ortho*sign(ortho[(arange(ortho.shape[0]),abs(ortho).argmax(axis=0))]).reshape(1,-1)
    numpy.testing.assert_allclose( ortho2, identity(3) )
    
    A = identity(3)
    A[0,0] = 10
    A[1,1] = -1
    A[2,2] = .00001
    ortho = orthonormalize( A )
    ## They should be the same now up to sign changes and permutations.
    ## Let's assume no permutations and flip signs to that the largest element
    ## in each column is positive.
    ortho2 = ortho*sign(ortho[(arange(ortho.shape[0]),abs(ortho).argmax(axis=0))]).reshape(1,-1)
    numpy.testing.assert_allclose( ortho2, identity(3) )

def test_canonical_point():
    p_closest = canonical_point( [1,1,0], array([[1,0,0],[0,1,0]]).T )
    numpy.testing.assert_allclose( p_closest, zeros(3) )
    
def main():
    import nose
    nose.runmodule()

if __name__ == '__main__':
    main()
