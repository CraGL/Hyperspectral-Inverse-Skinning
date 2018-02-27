from __future__ import division, print_function

import numpy
from numpy import *

def principal_cosangles( A, B, orthonormal = False ):
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
    principal_angles = linalg.svd( A.T.dot( B ), compute_uv = False )
    ## Any dimension mis-match should produce zero singular values.
    principal_angles = append(
        principal_angles,
        zeros(max(A.shape[1],B.shape[1])-min(A.shape[1],B.shape[1]))
        )
    return principal_angles

def principal_angles( A, B, orthonormal = False ):
    '''
    Given:
        A: A matrix whose columns are directions in the ambient space
        B: A matrix whose columns are directions in the ambient space
    Returns:
        The 1D array of principal angles (in radians) between A and B
    '''
    cosangles = principal_cosangles( A, B, orthonormal )
    ## Clip because acos() hates values epsilon outside [-1,1].
    angles = arccos( cosangles.clip(-1,1) )
    return angles

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

def optimal_p_given_B_for_flats_ortho( B, flats ):
    '''
    Given:
        B: A (not necessarily orthonormal) matrix whose columns are directions in the ambient space R^n.
        flats: A sequence of ( matrix, vector ) pairs ( A, a ) defining flats implicitly via A*(x-a)=0 for x in R^n.
    Returns:
        p: The point p in R^n for argmin_{p,z} sum_{A,a in flats} |A(p+Bz-a)|^2, (aka a point through which the explicit flat p + Bz should pass).
    '''
    
    ## E = |A(p+Bz-a)|^2 = M:M
    ## dE = 2M : dM
    ## dM = A dp + AB dz
    ## dE = 2M:A dp + 2M:A B dz
    ## dE = 2 A' M : dp + 2 B' A' M : dz
    ## dE/dz = 2 B' A' M = 2 B' A' (Ap + ABz - Aa) = 0
    ##      <=> 2 B' A' A B z = -2 B'A'(Ap-Aa)
    ##      <=> z = -inv(B' A' A B) B'A'(Ap-Aa)
    ## => M = Ap-Aa - AB inv(B' A' A B) B'A'(Ap-Aa)
    ##      = ( I - AB inv(B' A' A B) B'A' ) (Ap-Aa)
    ## Q = ( I - AB inv(B' A' A B) B'A' )
    ## Q is a projection matrix, so QQ = Q and Q' = Q
    ## M = Q (Ap-Aa)
    ## dM = Q A dp
    ## dE = 2M : dM = 2M : Q A dp
    ##    = 2 A' Q M : dp
    ## dE/dp = 0 = 2 A' Q M = 2 A' Q Q ( Ap-Aa ) = 2 A' Q A ( p - a )
    ##      <=> sum_i ( 2 A' Q A ) p = sum_i( 2 A' Q A a )
    
    assert len( flats ) > 0
    
    n = flats[0][0].shape[1]
    
    system = zeros( (n,n) )
    rhs = zeros( n )
    
    for A, a in flats:
        AB = dot( A, B )
        ## If A can have fewer rows than B has columns, then we need lstsq() to be safe.
        Q = -dot( AB, linalg.lstsq( dot( AB.T, AB ), AB.T )[0] )
        Q[diag_indices_from(Q)] += 1
        AQA = dot( A.T, dot( Q, A ) )
        system += AQA
        rhs += dot( AQA, a )
    
    ## The smallest singular value is always small, because any point on the flat
    ## is just as good as any other. Use lstsq() to find the smallest norm solution.
    # assert linalg.norm( system, ord = -2 ) < 1e-5
    p = linalg.lstsq( system, rhs )[0]
    return p

def test_principal_angles():
    A = array([[1,0,0]]).T
    B = array([[0,1,0]]).T
    angles = principal_angles( A, B, orthonormal = True )
    numpy.testing.assert_allclose( angles, pi/2*ones(1) )
    cosangles = principal_cosangles( A, B, orthonormal = True )
    numpy.testing.assert_allclose( cosangles, zeros(1) )
    
    ## Check if orthonormalization is working.
    angles = principal_angles( 2*A, -10*B )
    numpy.testing.assert_allclose( angles, pi/2*ones(1) )
    cosangles = principal_cosangles( 2*A, -10*B )
    numpy.testing.assert_allclose( cosangles, zeros(1) )
    
    ## When one flat has more dimensions than the other, there should be more 90-degree
    ## angles returned.
    A = array([[1,0,0],[0,0,1]]).T
    B = array([[0,1,0]]).T
    angles = principal_angles( A, B )
    numpy.testing.assert_allclose( angles, pi/2*ones(2) )
    cosangles = principal_cosangles( A, B )
    numpy.testing.assert_allclose( cosangles, zeros(2) )
    
    A = array([[1,1,0]]).T
    B = array([[1,0,0]]).T
    angles = principal_angles( A, B )
    numpy.testing.assert_allclose( angles, pi/4*ones(1) )
    cosangles = principal_cosangles( A, B )
    numpy.testing.assert_allclose( cosangles, (1/sqrt(2))*ones(1) )
    
    A = array([[1,0,0],[0,0,1]]).T
    B = array([[1,1,0]]).T
    angles = principal_angles( A, B )
    numpy.testing.assert_allclose( angles, [ pi/4, pi/2 ] )
    cosangles = principal_cosangles( A, B )
    numpy.testing.assert_allclose( cosangles, [ cos(pi/4), 0 ] )

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

def test_optimal_p_given_B_for_flats_ortho():
    dim = 5
    ortho_dim = 2
    p_truth = random.random(dim)
    flats = [ ( random.random((ortho_dim,dim)), p_truth ) for i in range(10) ]
    ## Make sure B dimension is <= ortho_dim dimension
    B = random.random((dim,2))
    p_best = optimal_p_given_B_for_flats_ortho( B, flats )
    numpy.testing.assert_allclose( p_best, p_truth )

def main():
    print( "Debug the following by running: python -m pytest --pdb flat_metrics.py" )
    import pytest
    pytest.main([__file__])
    # test_optimal_p_given_B_for_flats_ortho()

if __name__ == '__main__':
    main()
