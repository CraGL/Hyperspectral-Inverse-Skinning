"""
Sample code automatically generated on 2018-01-01 04:47:59

by www.matrixcalculus.org

from input

d/dz norm2(V*W*z-vprime)^2 = 2*W'*V'*(V*W*z-vprime)
d/dW norm2(V*W*z-vprime)^2 = 2*V'*(V*W*z-vprime)*z'

where

V is vbar (3p-by-12p)
vprime is a 3p-vector
W is a 12p-by-handles matrix
z is a vector of affine mixing weights for the columns of W (handles)

For the W'*V'*V*W matrix to be invertible, we need
    max(12p, handles, 3p) = max( 3p, handles ) >= handles
which is equivalent to:
    3p >= handles
(Just in case, we can use the pseudoinverse.)


The generated code is provided"as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

USE_PSEUDOINVERSE = True

## TODO: pack() and unpack() and A_from_non_Cayley_B() should use the Grassmann manifold parameters only
##       and zero the rest (or rotate appropriately).
def unpack( x, poses, handles ):
    W = x.reshape( 12*poses, handles )
    
    assert 12*poses - handles > 0
    
    return W

def pack( W, poses, handles ):
    return W.ravel()

def quadratic_for_z( W, V, vprime ):
    '''
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of `z`:
        energy = np.dot( np.dot( z, Q ), z ) + np.dot( L, Z ) + C
    '''
    
    assert len( W.shape ) == 2
    assert len( V.shape ) == 2
    assert len( vprime.shape ) == 1
    assert W.shape[0] == V.shape[1]
    assert V.shape[0] == vprime.shape[0]
    
    VW = np.dot( V, W )
    
    Q = np.dot( VW.T, VW )
    L = -2.0*np.dot( vprime, VW )
    C = np.dot( vprime, vprime )
    
    return Q, L, C

def solve_for_z( W, V, vprime, return_energy = False ):
    Q, L, C = quadratic_for_z( W, V, vprime )
    
    ## We also need the constraint that z.sum() == 1
    handles = len(L)
    Qbig = np.block( [ [ Q, np.ones((handles,1)) ], [np.ones((1,handles)), np.zeros((1,1)) ] ] )
    rhs = np.zeros( ( len(L) + 1 ) )
    rhs[:-1] = -0.5*L
    rhs[-1] = 1
    if USE_PSEUDOINVERSE:
        z = np.dot( np.linalg.pinv(Qbig), rhs )[:-1]
    else:
        z = np.linalg.solve( Qbig, rhs )[:-1]
    
    E = np.dot( np.dot( z, Q ), z ) + np.dot( L, z ) + C
    print( "New function value after solve_for_z():", E )
    if return_energy:
        return z, E
    else:
        return z

def linear_matrix_equation_for_W( V, vprime, z ):
    '''
    Returns (A, B, Y) to compute the gradient of the energy in terms of W
    in the following linear matrix equation:
        0.5 * dE/dW = np.dot( A, np.dot( W, B ) ) + Y
    '''
    
    assert len( V.shape ) == 2
    assert len( vprime.shape ) == 1
    assert len( z.shape ) == 1
    assert V.shape[0] == vprime.shape[0]
    
    ## Reshape the vectors into column matrices.
    vprime = vprime.reshape(-1,1)
    z = z.reshape(-1,1)
    
    A = np.dot( V.T, V )
    B = np.dot( z, z.T )
    Y = np.dot( np.dot( V.T, -vprime ), z.T )
    
    return A, B, Y

def solve_for_W( As, Bs, Ys ):
    assert len( As ) == len( Bs )
    assert len( As ) == len( Ys )
    assert len( As ) > 0
    
    system = np.zeros( ( Bs[0].shape[1]*As[0].shape[0], Bs[0].shape[0]*As[0].shape[1] ) )
    ## Our kronecker product formula assumes column-major vectorization.
    ## In that case, the identity is: vec( A*X*B ) = kron( A, B.T ) * vec(X)
    rhs = np.zeros( Ys[0].ravel().shape )
    for A, B, Y in zip( As, Bs, Ys ):
        system += np.kron( A, B.T )
        rhs -= Y.ravel()
    
    if USE_PSEUDOINVERSE:
        W = np.dot( np.linalg.pinv(system), rhs ).reshape( A.shape[0], B.shape[1] )
    else:
        W = np.linalg.solve( system, rhs ).reshape( A.shape[0], B.shape[1] )
    
    ## Normalize the columns of W.
    columns_norm2 = ( W*W ).sum( axis = 0 )
    assert columns_norm2.min() > 1e-10
    W /= np.sqrt( columns_norm2 ).reshape( 1, -1 )
    
    return W

def generateRandomData( poses = None, handles = None ):
    # np.random.seed(0)
    
    ## If this isn't true, the inv() in the energy will fail.
    assert 3*poses >= handles or USE_PSEUDOINVERSE
    
    W = np.random.randn(12*poses, handles)
    V = np.random.randn(3*poses, 12*poses)
    vprime = np.random.randn(3*poses)
    return W, V, vprime, poses, handles

if __name__ == '__main__':
    USE_PSEUDOINVERSE = False
    
    W, V, vprime, poses, handles = generateRandomData( poses = 2, handles = 5 )
    
    z, f = solve_for_z( W, V, vprime, return_energy = True )
    
    import flat_intersection_direct_gradients
    f2, _, _ = flat_intersection_direct_gradients.fAndGpAndHp_fast( W[:,0], W[:,1:] - W[:,:1], V, vprime )
    
    print( 'function value:', f )
    print( 'other function value:', f2 )
    print( '|function difference|:', abs( f - f2 ) )
    
    A, B, Y = linear_matrix_equation_for_W( V, vprime, z )
    W_next = solve_for_W( [A], [B], [Y] )
    print( 'W:', W )
    print( 'W from solve_for_W():', W_next )
    print( 'W from solve_for_W() column norms:', ( W_next*W_next ).sum(0) )
    print( '|W difference|:', abs( W - W_next ).max() )
    
    x = pack( W, poses, handles )
    W2 = unpack( W, poses, handles )
    x2 = pack( W2, poses, handles )
    print( "If pack/unpack work, these should be zeros:" )
    print( abs( W - W2 ).max() )
    print( abs( x - x2 ).max() )
