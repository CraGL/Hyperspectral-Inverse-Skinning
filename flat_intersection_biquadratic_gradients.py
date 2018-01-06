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

For the formulation finding a V which minimizes the expression,
we de-vectorize W*z into a stack of T 3x3 matrices and t translations and substitute
b=t-vprime:

d/dv norm2(T*v+b)^2 = 2*T'*(b+T*v)

where

T is a 3px3 matrix
b is a 3p-vector
v id a 3p-vector

The generated code is provided"as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

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
        energy = np.dot( np.dot( z, Q ), z ) + np.dot( L, z ) + C
    '''
    
    vprime = vprime.squeeze()
    
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

def solve_for_z( W, V, vprime, return_energy = False, use_pseudoinverse = True ):
    Q, L, C = quadratic_for_z( W, V, vprime )
    
    sv = np.linalg.svd( Q, compute_uv = False )
    smallest_singular_value = sv[-1]
    #if sv[-1] < 1e-5:
    #    print( "Vertex has small singular values:", sv )
    #    return ( None, 0.0 ) if return_energy else None
    
    ## We also need the constraint that z.sum() == 1
    handles = len(L)
    
    ## numpy.block() is extremely slow:
    # Qbig = np.block( [ [ Q, np.ones((handles,1)) ], [np.ones((1,handles)), np.zeros((1,1)) ] ] )
    ## This is the same but much faster:
    Qbig = np.zeros( (Q.shape[0]+1, Q.shape[1]+1) )
    Qbig[:-1,:-1] = Q
    Qbig[-1,:-1] = 1
    Qbig[:-1,-1] = 1
    
    rhs = np.zeros( ( len(L) + 1 ) )
    rhs[:-1] = -0.5*L
    rhs[-1] = 1
    if use_pseudoinverse:
        z = np.dot( np.linalg.pinv(Qbig), rhs )[:-1]
    else:
        z = np.linalg.solve( Qbig, rhs )[:-1]
    
    ## This always passes:
    # assert abs( z.sum() - 1.0 ) < 1e-10
    
    if return_energy:
        E = np.dot( np.dot( z, Q ), z ) + np.dot( L, z ) + C
        # print( "New function value after solve_for_z():", E )
        return z, smallest_singular_value, E
    else:
        return z, smallest_singular_value

def quadratic_for_V( W, z, vprime ):
    '''
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of the 3-vector
    `v`, the rest pose position which are converted to V via:
        kron( identity(poses), kron( identity(3), append( v, [1] ).reshape(1,-1) ) )
        =
        kron( identity(poses*3), append( v, [1] ).reshape(1,-1) )
    
    The quadratic expression returned is:
        energy = np.dot( np.dot( v, Q ), v ) + np.dot( L, v ) + C
    '''
    
    z = z.squeeze()
    vprime = vprime.squeeze()
    
    assert len( W.shape ) == 2
    assert len( z.shape ) == 1
    assert len( vprime.shape ) == 1
    assert W.shape[1] == z.shape[0]
    
    Taffine = np.dot( W, z ).reshape( -1,4 )
    ## It should be a horizontal stack of poses 3-by-4 matrices.
    assert Taffine.shape[0] % 3 == 0
    ## Separate the left 3x3 from the translation
    T = Taffine[:,:3]
    t = Taffine[:,3]
    
    b = t - vprime
    assert len( b.shape ) == 1
    
    Q = np.dot( T.T, T )
    L = 2.0*np.dot( T.T, b )
    C = np.dot( b, b )
    
    return Q, L, C

def solve_for_V( W, z, vprime, return_energy = False, use_pseudoinverse = False ):
    Q, L, C = quadratic_for_V( W, z, vprime )
    
    if use_pseudoinverse:
        v = np.dot( np.linalg.pinv(Q), -0.5*L )
    else:
        v = np.linalg.solve( Q, -0.5*L )
    
    ## Restore V to a matrix.
    assert len(vprime) % 3 == 0
    poses = len(vprime)//3
    # V = np.kron( np.identity(poses), np.kron( np.identity(3), np.append( v, [1] ).reshape(1,-1) ) )
    V = np.kron( np.identity(poses*3), np.append( v, [1] ).reshape(1,-1) )
    
    if return_energy:
        E = np.dot( np.dot( v, Q ), v ) + np.dot( L, v ) + C
        # print( "New function value after solve_for_V():", E )
        return V, E
    else:
        return V

def linear_matrix_equation_for_W( V, vprime, z ):
    '''
    Returns (A, B, Y) to compute the gradient of the energy in terms of W
    in the following linear matrix equation:
        0.5 * dE/dW = np.dot( A, np.dot( W, B ) ) + Y
    '''
    
    vprime = vprime.squeeze()
    z = z.squeeze()
    
    assert len( V.shape ) == 2
    assert len( vprime.shape ) == 1
    assert len( z.shape ) == 1
    assert V.shape[0] == vprime.shape[0]
    
    ## Reshape the vectors into column matrices.
    vprime = vprime.reshape(-1,1)
    z = z.reshape(-1,1)
    
    # A = V'*V = ( I_3poses kron [v 1] )'*( I_3poses kron [v 1] ) = ( I_3poses kron [v 1]' )*( I_3poses kron [v 1] ) = I_3poses kron ( [v 1]'*[v 1] )
    # B' = B = z*z' = z kron z' = z kron z'
    # A kron B' = ( I_3poses kron ( [v 1]'*[v 1] ) ) kron ( z kron z' ) = I_3poses kron( ( [v 1]'*[v 1] ) kron ( z*z' ) )
    v = V[0,:4].reshape(1,-1)
    A = np.outer( v.T, v ) #, np.dot( V.T, V )
    B = np.dot( z, z.T )
    Y = np.dot( np.dot( V.T, -vprime ), z.T )
    
    return A, B, Y

def solve_for_W( As, Bs, Ys, use_pseudoinverse = True ):
    assert len( As ) == len( Bs )
    assert len( As ) == len( Ys )
    assert len( As ) > 0
    
    assert Ys[0].shape[0] % 12 == 0
    poses = Ys[0].shape[0]//12
    system = np.zeros( ( Bs[0].shape[1]*As[0].shape[0], Bs[0].shape[0]*As[0].shape[1] ) )
    ## Our kronecker product formula assumes column-major vectorization.
    ## In that case, the identity is: vec( A*X*B ) = kron( A, B.T ) * vec(X)
    ## Since the system matrix is a repeated block diagonal, we can just store
    ## the block and solve against the right-hand-side reshaped with each N entries as
    ## a column.
    ## Vectorize (ravel) the right-hand side at the end.
    rhs = np.zeros( Ys[0].shape )
    for A, B, Y in zip( As, Bs, Ys ):
        ## There is no point to doing this, since the inverse of a block diagonal matrix
        ## is the inverse of each block (and these blocks are repeated).
        # system += np.kron( np.eye( 3*poses ), np.kron( A[0], B.T ) )
        system += np.kron( A, B.T )
        # system += np.kron( A[1], B.T )
        rhs -= Y
    
    # import scipy.linalg
    # system_big = scipy.linalg.block_diag( *( [system]*(3*poses) ) )
    
    if use_pseudoinverse:
        # W1 = np.dot( np.linalg.pinv(system_big), rhs.ravel() ).reshape( 12*poses, B.shape[1] )
        W = np.dot( np.linalg.pinv(system), rhs.reshape( -1, system.shape[0] ).T ).T.reshape( 12*poses, B.shape[1] )
        # print( "pinv block difference:", abs( W - W1 ).max() )
    else:
        # W1 = np.linalg.solve( system_big, rhs.ravel() ).reshape( 12*poses, B.shape[1] )
        W = np.linalg.solve( system, rhs.reshape( -1, system.shape[0] ).T ).T.reshape( 12*poses, B.shape[1] )
        # print( "solve block difference:", abs( W - W1 ).max() )
    
    ## Normalize the columns of W.
    ## UPDATE: No, this is wrong. W's columns are points, not vectors.
    # columns_norm2 = ( W*W ).sum( axis = 0 )
    # assert columns_norm2.min() > 1e-10
    # W /= np.sqrt( columns_norm2 ).reshape( 1, -1 )
    
    ## We could normalize the difference from the average, but it's always large:
    # print( "W column norm:", ( ( W - np.average( W, axis = 1 ).reshape(-1,1) )**2 ).sum(axis=0) )
    
    return W

def generateRandomData( poses = None, handles = None ):
    # np.random.seed(0)
    
    ## If this isn't true, the inv() in the energy will fail.
    if 3*poses < handles:
        print( "You'd better use the pseudoinverse or you'll get unpredictable results." )
    
    W = np.random.randn(12*poses, handles)
    V = np.random.randn(3*poses, 12*poses)
    vprime = np.random.randn(3*poses)
    return W, V, vprime, poses, handles

if __name__ == '__main__':
    use_pseudoinverse = False
    
    W, V, vprime, poses, handles = generateRandomData( poses = 1, handles = 5 )
    
    z, ssv, f = solve_for_z( W, V, vprime, return_energy = True, use_pseudoinverse = use_pseudoinverse )
    
    import flat_intersection_direct_gradients
    f2, _, _ = flat_intersection_direct_gradients.fAndGpAndHp_fast( W[:,0], W[:,1:] - W[:,:1], V, vprime )
    
    print( 'function value:', f )
    print( 'other function value:', f2 )
    print( '|function difference|:', abs( f - f2 ) )
    
    A, B, Y = linear_matrix_equation_for_W( V, vprime, z )
    W_next = solve_for_W( [A], [B], [Y], use_pseudoinverse = use_pseudoinverse )
    print( 'W:', W )
    print( 'W from solve_for_W():', W_next )
    # print( 'W from solve_for_W() column norms:', ( W_next*W_next ).sum(0) )
    print( '|W difference|:', abs( W - W_next ).max() )
    
    x = pack( W, poses, handles )
    W2 = unpack( W, poses, handles )
    x2 = pack( W2, poses, handles )
    print( "If pack/unpack work, these should be zeros:" )
    print( abs( W - W2 ).max() )
    print( abs( x - x2 ).max() )
