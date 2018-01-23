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

def quadratic_for_z( W, v, vprime ):
    '''
    Given:
    	W: The 12*P-by-handles array
    	v: an array [ x y z 1 ]
    	vprime: a 3*P array of all deformed poses
    
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of `z`:
        energy = np.dot( np.dot( z, Q ), z ) + np.dot( L, z ) + C
    '''
    
    v = v.squeeze()
    vprime = vprime.squeeze()
    
    assert len( W.shape ) == 2
    assert v.shape == (4,)
    assert len( vprime.shape ) == 1
    assert W.shape[0] % 12 == 0
    assert vprime.shape[0] % 3 == 0
    assert vprime.shape[0]*4 == W.shape[0]
    
    # V = np.kron( np.identity(vprime.shape[0]), v.reshape(1,-1) )
    # VW = np.dot( V, W )
    # abs( np.dot( v, W.T.reshape( -1, 4 ).T ).reshape( W.shape[1], -1 ).T - V.dot(W) ).max()
    
    VW = np.dot( v, W.T.reshape( -1, 4 ).T ).reshape( W.shape[1], -1 ).T
    
    Q = np.dot( VW.T, VW )
    L = -2.0*np.dot( vprime, VW )
    C = np.dot( vprime, vprime )
    
    return Q, L, C

def solve_for_z( W, v, vprime, return_energy = False, use_pseudoinverse = True, strategy = None, **kwargs ):
    Q, L, C = quadratic_for_z( W, v, vprime )
    
    # sv = np.linalg.svd( Q, compute_uv = False )
    # smallest_singular_value = sv[-1]
    smallest_singular_value = np.linalg.norm( Q, ord = -2 )
    # print( smallest_singular_value - sv2 )
    #if smallest_singular_value < 1e-5:
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
    
    if strategy not in (None, 'positive', 'sparse4', 'neighbors'):
        raise RuntimeError( "Unknown strategy: " + repr(strategy) )
    
    if strategy == 'neighbors' and 'neighborz' in kwargs:
        neighborz = kwargs['neighborz']
        neighbor_weight = kwargs['neighbor_weight']
        ## Add the identity * weight to the original diagonal.
        Qbig[ np.diag_indices( Q.shape[0] ) ] += neighbor_weight
        rhs[ :Q.shape[0] ] += neighbor_weight*neighborz
    
    if strategy == 'positive':
        assert not use_pseudoinverse
        import cvxopt.solvers
        bounds_system = cvxopt.matrix( -np.eye(len(L)) )
        bounds_rhs = cvxopt.matrix( np.zeros( ( len(L), 1 ) ) )
        eq_system = cvxopt.matrix( np.ones( ( 1, len(L) ) ) )
        eq_rhs = cvxopt.matrix( np.ones( ( 1, 1 ) ) )
        z = np.array( cvxopt.solvers.qp(
            cvxopt.matrix( Q ), cvxopt.matrix( np.zeros( (len(L),1) ) ),
            bounds_system, bounds_rhs, eq_system, eq_rhs,
            options = {'show_progress': False}
            )['x'] ).squeeze()
        # print( 'z:', z )
    elif use_pseudoinverse:
        z = np.dot( np.linalg.pinv(Qbig), rhs )[:-1]
    else:
        z = np.linalg.solve( Qbig, rhs )[:-1]
    
    if strategy == 'sparse4':
        ## Constrain all but the 4 biggest weights to 0 and re-solve.
        ## UPDATE: Use the absolute value of the weights. Those are the closest to "unused".
        # biggest_z_indices = np.argsort(z)[:-4]
        biggest_z_indices = np.argsort(np.abs(z))[:-4]
        ## It only makes sense to do this if there are at least 4 handles:
        if len(biggest_z_indices) > 0:
            ## Constrain them to 0's.
            rhs[ biggest_z_indices ] = 0
            ## Zero the rows and columns. (We can zero the columns like this
            ## because the corresponding right-hand-side values are 0.)
            ## (Zeroing the columns keeps the system symmetric.)
            Qbig[ biggest_z_indices, : ] = 0
            Qbig[ :, biggest_z_indices ] = 0
            ## Set the diagonal to the identity matrix.
            ## NOTE: This tuple() is very important. Passing a numpy.array has a
            ##       very different result.
            Qbig[ tuple(np.tile( biggest_z_indices.reshape(1,-1), (2,1) )) ] = 1.
            ## Re-solve.
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

def quadratic_for_v( W, z, vprime ):
    '''
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of the 3-vector
    `v`, the rest pose position which can be converted to V bar via:
        kron( identity(poses), kron( identity(3), append( v, [1] ).reshape(1,-1) ) )
        =
        kron( identity(poses*3), append( v, [1] ).reshape(1,-1) )
    
    The quadratic expression returned is:
        energy = np.dot( np.dot( v[:3], Q ), v[:3] ) + np.dot( L, v[:3] ) + C
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

def solve_for_v( W, z, vprime, return_energy = False, use_pseudoinverse = False ):
    Q, L, C = quadratic_for_v( W, z, vprime )
    
    if use_pseudoinverse:
        v = np.dot( np.linalg.pinv(Q), -0.5*L )
    else:
        v = np.linalg.solve( Q, -0.5*L )
    
    ## Restore V to a matrix.
    assert len(vprime) % 3 == 0
    poses = len(vprime)//3
    # V = np.kron( np.identity(poses), np.kron( np.identity(3), np.append( v, [1] ).reshape(1,-1) ) )
    # V = np.kron( np.identity(poses*3), np.append( v, [1] ).reshape(1,-1) )
    result_v = np.append( v, [1] )
    
    if return_energy:
        E = np.dot( np.dot( v, Q ), v ) + np.dot( L, v ) + C
        # print( "New function value after solve_for_V():", E )
        return V, E
    else:
        return V

def linear_matrix_equation_for_W( v, vprime, z ):
    '''
    Returns (A, B, Y) to compute the gradient of the energy in terms of W
    in the following linear matrix equation:
        0.5 * dE/dW = np.dot( A, np.dot( W, B ) ) + Y
    '''
    
    v = v.squeeze()
    vprime = vprime.squeeze()
    z = z.squeeze()
    
    assert v.shape == (4,)
    assert len( vprime.shape ) == 1
    assert len( z.shape ) == 1
    
    ## Reshape the vectors into column matrices.
    vprime = vprime.reshape(-1,1)
    z = z.reshape(-1,1)
    
    # A = V'*V = ( I_3poses kron [v 1] )'*( I_3poses kron [v 1] ) = ( I_3poses kron [v 1]' )*( I_3poses kron [v 1] ) = I_3poses kron ( [v 1]'*[v 1] )
    # B' = B = z*z' = z kron z' = z kron z'
    # A kron B' = ( I_3poses kron ( [v 1]'*[v 1] ) ) kron ( z kron z' ) = I_3poses kron( ( [v 1]'*[v 1] ) kron ( z*z' ) )
    v = v.reshape(1,-1)
    A = np.outer( v.T, v ) #, np.dot( V.T, V )
    B = np.dot( z, z.T )
    ## We can also get simplify the block diagonal V multiplication.
    # Y = np.dot( np.dot( V.T, -vprime ), z.T )
    Y = np.dot( np.dot( (-v).T, vprime.T ).T.reshape(-1,1), z.T )
    
    return A, B, Y

def zero_system_for_W( A, B, Y ):
    system = np.zeros( ( B.shape[1]*A.shape[0], B.shape[0]*A.shape[1] ) )
    rhs = np.zeros( Y.shape )
    
    return system, rhs
    
def accumulate_system_for_W( system, rhs, A, B, Y, weight ):
    system += np.kron( weight*A, B.T )
    rhs -= weight*Y

def solve_for_W( As, Bs, Ys, use_pseudoinverse = True, projection = None, **kwargs ):
    assert len( As ) == len( Bs )
    assert len( As ) == len( Ys )
    assert len( As ) > 0
    
    assert Ys[0].shape[0] % 12 == 0
    poses = Ys[0].shape[0]//12
    system = np.zeros( ( Bs[0].shape[1]*As[0].shape[0], Bs[0].shape[0]*As[0].shape[1] ) )
    ## Our kronecker product formula is the one for row-major vectorization:
    ##     vec( A*X*B ) = kron( A, B.T ) * vec(X)
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
    
    return solve_for_W_with_system_rhs( system, rhs, use_pseudoinverse = use_pseudoinverse, projection = projection, **kwargs )

def solve_for_W_with_system( system, rhs, use_pseudoinverse = True, projection = None, **kwargs ):
    assert rhs.shape[0] % 12 == 0
    poses = rhs.shape[0]//12
    assert system.shape[0] % 4 == 0
    handles = system.shape[0] // 4
    
    if projection == 'regularize_translation':
        ## Let's add a small weight penalizing non-zero values for the elements
        ## of W corresponding to translations.
        assert system.shape[0] == 4*handles
        reg = np.zeros( system.shape[0] )
        reg[ 3*handles : 4*handles ] = 1e-3 # 1.0
        system[ np.diag_indices_from(system) ] += reg
    elif projection == 'regularize_identity':
        ## Let's add a small weight penalizing non-zero values for the elements
        ## of W corresponding to translations.
        assert system.shape[0] == 4*handles
        w = 1e-3
        reg = w*np.ones( system.shape[0] )
        system[ np.diag_indices_from(system) ] += reg
        assert rhs.shape[0] == 12*poses
        assert rhs.shape[1] == handles
        identity = np.tile( w*np.eye(4)[:3].ravel(), poses ).reshape(-1,1)
        rhs += identity
    
    # import scipy.linalg
    # system_big = scipy.linalg.block_diag( *( [system]*(3*poses) ) )
    
    if projection == 'first':
        assert rhs.shape == ( 12*poses, handles )
        ## Add a least squares term setting the first column of the output
        ## to kwargs['first_column'].
        ## For the full system matrix, that would be (using a pre-vectorized rhs):
        ##     system += w_lsq * diag([ 1, 0 repeated h-1 times, 1, 0 repeated h-1 times, ... ])
        ##     rhs[:,0] += w_lsq * first_column
        ## where h is the number of handles, aka B.shape[0] or B.shape[1].
        ## For the small system matrix, we just need the first 4 repetitions.
        
        ## For debugging, keep the old system and rhs around.
        system_orig = system.copy()
        rhs_orig = rhs.copy()
        
        first_column = kwargs['first_column']
        w_lsq = 1e4
        rhs[:,0] += w_lsq * first_column
        ## Modify diagonal of system
        h = handles
        system[0*h,0*h] += w_lsq
        system[1*h,1*h] += w_lsq
        system[2*h,2*h] += w_lsq
        system[3*h,3*h] += w_lsq
        
        ## Now we will fall through to the system solving.
    
    if projection == 'constrain_magnitude':
        assert not use_pseudoinverse
        import cvxopt.solvers
        n = system.shape[0]
        bounds_system = cvxopt.matrix( np.vstack( [ np.eye(n)[:3*handles], -np.eye(n)[:3*handles] ] ) )
        bounds_rhs = cvxopt.matrix( np.ones( ( 2*3*handles, 1 ) ) )
        W = [ np.array( cvxopt.solvers.qp( cvxopt.matrix( system ), cvxopt.matrix( col.reshape(-1,1) ), bounds_system, bounds_rhs )['x'] ) for col in rhs.reshape( -1, n ) ]
        W = np.hstack(W).T.reshape( 12*poses, handles )
    elif use_pseudoinverse:
        # W1 = np.dot( np.linalg.pinv(system_big), rhs.ravel() ).reshape( 12*poses, handles )
        W = np.dot( np.linalg.pinv(system), rhs.reshape( -1, system.shape[0] ).T ).T.reshape( 12*poses, handles )
        # print( "pinv block difference:", abs( W - W1 ).max() )
    else:
        # W1 = np.linalg.solve( system_big, rhs.ravel() ).reshape( 12*poses, handles )
        ## With our numpy solver:
        # W = np.linalg.solve( system, rhs.reshape( -1, system.shape[0] ).T ).T.reshape( 12*poses, handles )
        ## We know that the system is symmetric positive definite because it is the
        ## sum of the kronecker product of the outer product of vectors.
        ## Outer products of vectors are positive definite, and sum and kronecker products
        ## preserve positive semidefiniteness.
        ## So we can use scipy.linalg.solve() which can take advantage of that fact.
        import scipy.linalg
        W = scipy.linalg.solve( system, rhs.reshape( -1, system.shape[0] ).T, sym_pos = True ).T.reshape( 12*poses, handles )
        # print( "solve block difference:", abs( W - W1 ).max() )
    
    ## Normalize the columns of W.
    ## UPDATE: No, this is wrong. W's columns are points, not vectors.
    # columns_norm2 = ( W*W ).sum( axis = 0 )
    # assert columns_norm2.min() > 1e-10
    # W /= np.sqrt( columns_norm2 ).reshape( 1, -1 )
    
    ## We could normalize the difference from the average:
    print( "W column norm:", ( ( W - np.average( W, axis = 1 ).reshape(-1,1) )**2 ).sum(axis=0) )
    ## Actually, it grows. Let's stop that.
    if projection == 'normalize':
        avg = np.average( W, axis = 1 ).reshape(-1,1)
        B = W - avg
        
        ## Simple: normalize the magnitude.
        B /= np.sqrt( ( B*B ).sum( axis = 0 ) ).reshape(1,-1)
        ## Fancy: Grassmann projection
        ## UPDATE: This doesn't work because it is unstable, so x never falls below its threshold.
        # import flat_intersection_cayley_grassmann_gradients as grassmann
        # grassmann_params = grassmann.A_from_non_Cayley_B( B )
        # B = grassmann.B_from_Cayley_A( grassmann_params )
        
        W = avg + B
    elif projection == 'first' and False: # This causes things to go crazy.
        ## First make the first column of W exact.
        W[:,0] = first_column
        ## Then normalize against the first column.
        B = W[:,1:] - W[:,:1]
        ## Normalize the magnitude.
        B /= np.sqrt( ( B*B ).sum( axis = 0 ) ).reshape(1,-1)
        ## Put it back.
        W[:,1:] = B
    elif projection in ( 'regularize_translation', 'regularize_identity', 'constrain_magnitude' ):
        ## We handled this above.
        pass
    elif projection is not None:
        raise RuntimeError( "Unknown projection: " + repr(projection) )
    
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
