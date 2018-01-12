"""
E_local_bad = \sum_i | \bar{V}_i*( t_i - \sum_{j \in N(i)} w_ij t_j ) |^2
    E_local_bad will likely always be locally linear in 3D. We want it to be locally linear
in 12D. Drop the \bar{V}_i metric.

E_local = \sum_i | t_i - \sum_{j \in N(i)} w_ij t_j |^2
E_data = \sum_i | \bar{V}_i*t_i - vprime |^2 = \sum_i | (I_3p kron [v 1])*t_i - vprime |^2
       = \sum_i t_i'*( I_3poses kron ( [v 1]'*[v 1] ) )*t_i - 2*vprime'*(I_3p kron [v 1])*t_i + vprime'*vprime

where

V is vbar (3p-by-12p) = kron( identity(poses*3), append( v, [1] ).reshape(1,-1) )
t_i is a 12p vector formed by vectorizing vertex i's transformations
vprime is a 3p-vector
w_ij are scalar weights


E_local = \sum_i | T L_i' |^2 = \sum_i | ( I_12p kron L_i ) vec_rowmajor( T ) |^2 = | I_12p kron L vec_rowmajor( T ) |^2
        = \sum_i | ( L_i kron I_12p ) vec_colmajor( T ) |^2
        = | ( L kron I_12p ) vec_colmajor( T ) |^2 = vec_colmajor( T )' * ( ( L'*L ) kron I_12p ) * vec_colmajor( T )

where

L_i is a row of the #vertices-by-#vertices laplacian matrix L with neighbor weights w_ij (and w_ii = -1).
T is a 12p-by-#vertices matrix containing all the t_i columns side-by-side horizontally.

When solving for weights w_ij, only E_local is relevant (E_data is constant):

E_local = \sum_i | \bar{V}_i [ t_i t_j ... ] [ -1 w_ij ... ]' |^2 = \sum_i | [ t_j ... ] [ w_ij ... ]' - t_i |^2
    s.t. \sum_j w_ij = 1

where

[ t_j ... ] is a 12p-by-#neighbors matrix containing all the t_j columns of vertex i's neighbors' transformations.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.sparse

def quadratic_for_w( t_i, T_js ):
    '''
    Given:
        t_i: A 12p column vector of the transform for vertex i
        T_js: A 12p-by-#neighbors matrix where each column is the transform for a neighbor of vertex i.
    
    Returns a quadratic expression ( Q, L, C ) for the energy E_local in terms of `w`:
        energy = np.dot( np.dot( w, Q ), w ) + np.dot( L, w ) + C
    '''
    
    assert len( t_i.shape ) == 1
    assert len( T_js.shape ) == 2
    assert T_js.shape[0] == t_i.shape[0]
    
    Q = np.dot( T_js.T, T_js )
    L = -2.0*np.dot( t_i, T_js )
    C = np.dot( t_i, t_i )
    
    return Q, L, C

def solve_for_w( t_i, T_js, return_energy = False, strategy = None ):
    '''
    Given:
        t_i: A 12p column vector of the transform for vertex i
        T_js: A 12p-by-#neighbors matrix where each column is the transform for a neighbor of vertex i.
    
    Returns `w`, a #neighbors vector of weights (which sum to 1) for averaging the T_js.
    '''
    
    assert len( t_i ) % 12 == 0
    assert T_js.shape[0] == len( t_i )
    
    Q, L, C = quadratic_for_w( t_i, T_js )
    
    use_pseudoinverse = False
    smallest_singular_value = np.linalg.norm( Q, ord = -2 )
    if smallest_singular_value < 1e-5:
        print( "Vertex has small singular values (will use pseudoinverse):", np.linalg.svd( Q, compute_uv = False ) )
        # return ( None, 0.0 ) if return_energy else None
        use_pseudoinverse = True
    
    ## We also need the constraint that w.sum() == 1
    handles = len(L)
    
    ## numpy.block() is extremely slow:
    # Qbig = np.block( [ [ Q, np.ones((handles,1)) ], [np.ones((1,handles)), np.zeros((1,1)) ] ] )
    ## This is the same but much faster:
    Qbig = np.zeros( (handles+1, handles+1) )
    Qbig[:-1,:-1] = Q
    Qbig[-1,:-1] = 1
    Qbig[:-1,-1] = 1
    
    rhs = np.zeros( ( handles + 1 ) )
    rhs[:-1] = -0.5*L
    rhs[-1] = 1
    
    if strategy not in (None, 'positive'):
        raise RuntimeError( "Unknown strategy: " + repr(strategy) )
    
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
        w = np.dot( np.linalg.pinv(Qbig), rhs )[:-1]
    else:
        w = np.linalg.solve( Qbig, rhs )[:-1]
    
    ## This always passes:
    # assert abs( z.sum() - 1.0 ) < 1e-10
    
    if return_energy:
        E = np.dot( np.dot( w, Q ), w ) + np.dot( L, w ) + C
        # print( "New function value after solve_for_w():", E )
        return w, smallest_singular_value, E
    else:
        return w, smallest_singular_value

def quadratic_for_E_data( vs, vprimes ):
    '''
    Given:
        vs: The sequence of undeformed [ x y z ] positions
        vprimes: The sequence of 3*p [ x1 y1 z1 x2 y2 z2 x3 y3 z3 ] deformed positions, one per pose.
    
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of each 3-vector in
    `vs`, the rest pose positions which are converted to V via:
        kron( identity(poses*3), append( v, [1] ).reshape(1,-1) )
    
    The quadratic expression returned is (Q is sparse):
        energy = np.dot( T.ravel(order='F'), Q.dot( T.ravel(order='F') ) ) + np.dot( L, T.ravel(order='F') ) + C
    '''
    
    # E_data = \sum_i t_i'*( I_3poses kron ( [v 1]'*[v 1] ) )*t_i - 2*t_i*(I_3p kron [v 1]')*vprime + vprime'*vprime
    
    assert len( vs[0].shape ) == 1
    assert len( vprimes[0].shape ) == 1
    assert len( vs ) == len( vprimes )
    
    assert vprimes[0].shape[0] % 3 == 0
    poses = vprimes[0].shape[0] // 3
    
    num_vertices = vs.shape[0]
    
    block_diags = []
    rhs = np.zeros( num_vertices * 12 * poses )
    constant = 0.
    
    for i, v in enumerate(vs):
        v = np.append( v.squeeze(), [1] ).reshape(-1,1)
        vouter = np.dot( v, v.T )
        assert vouter.shape[0] == 4
        assert vouter.shape[1] == 4
        vprime = vprimes[i]
        
        block_diags.append( scipy.sparse.kron( scipy.sparse.eye(3*poses), vouter ) )
        
        rhs[ i*12*poses : (i+1)*12*poses ] = np.dot( np.kron( np.eye( 3*poses ), -2*v ), vprime.reshape(-1,1) ).squeeze()
        # rhs[ i*12*poses : (i+1)*12*poses ] = np.dot( np.tile(-2*v,(3*poses,1)), vprime.reshape(-1,1) ).squeeze()
        
        constant += np.dot( vprime, vprime )
    
    Q = scipy.sparse.block_diag( block_diags )
    L = rhs
    C = constant
    
    return Q, L, C

def evaluate_E_data( QLC, T ):
    '''
    Given:
        QLC: The return value of quadratic_for_E_data().
        T: A 12p-by-#vertices matrix where each column is the transform for a neighbor of vertex i.
    
    Returns the energy.
    '''
    
    assert len( T.shape ) == 2
    assert T.shape[0] % 12 == 0
    
    Q, L, C = QLC
    return np.dot( T.ravel(order='F'), Q.dot( T.ravel(order='F') ) ) + np.dot( L, T.ravel(order='F') ) + C

def quadratic_for_E_local( neighbors, ws, poses ):
    '''
    Given:
        neighbors: A sequence of indices for the neighbors of each element.
        ws:        A sequence of weights that sum to 1.0 corresponding to the neighbor at the corresponding index.
    
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of the
    12p transformation vectors t_i stacked vertically end-to-end.
    (If they are the columns of a matrix, it is that matrix vectorized in column-major order.)
    
    The quadratic expression returned is (Q is sparse):
        energy = np.dot( T.ravel(order='F'), Q.dot( T.ravel(order='F') ) )
    '''
    
    assert len( neighbors ) == len( ws )
    N = len( neighbors )
    
    ## Make room for the diagonal elements plus an element for everything in neighbors/ws.
    ijs = np.zeros( ( 2, N + np.sum([ len(neighs) for neighs in neighbors ]) ), dtype=int )
    vals = np.zeros( ijs.shape[1] )
    
    count = 0
    for i, neigh_i in enumerate( neighbors ):
        assert len( neigh_i ) == len( ws[i] )
        
        assert i not in neigh_i
        
        ## The diagonal element is always -1
        ijs[ :, count ] = i
        vals[ count ] = -1.0
        
        count += 1
        
        ## The next elements are the indices and values for the neighbors of i
        ijs[ 0, count : count + len( neigh_i ) ] = i
        ijs[ 1, count : count + len( neigh_i ) ] = neigh_i
        vals[ count : count + len( neigh_i ) ] = ws[i]
        
        count += len( neigh_i )
    
    assert count == len( vals )
    
    L = scipy.sparse.coo_matrix( ( vals, ijs ), shape = ( N, N ) )
    LTL = L.T.dot(L)
    
    Q = scipy.sparse.block_diag( [LTL]*(12*poses) )
    
    return Q

def evaluate_E_local( Q, T ):
    '''
    Given:
        Q: The return value of quadratic_for_E_local().
        T: A 12p-by-#vertices matrix where each column is the transform for a neighbor of vertex i.
    
    Returns the energy.
    '''
    
    assert len( T.shape ) == 2
    assert T.shape[0] % 12 == 0
    
    return np.dot( T.ravel(order='F'), Q.dot( T.ravel(order='F') ) )

def solve_for_T( E_data_quadratic, E_local_quadratic, poses ):
    '''
    Given:
        E_data_quadratic: The return value of quadratic_for_E_data().
        E_local_quadratic: The return value of quadratic_for_E_local().
        poses: The number of poses p
    
    Returns
        T: A 12p-by-#vertices matrix where each column is the transform for a neighbor of vertex i.
    '''
    
    import cvxopt, cvxopt.cholmod
    
    # system = E_data_quadratic[0].tocoo() + E_local_quadratic.tocoo()
    # system = cvxopt.spmatrix( system.data, np.asarray( system.row, dtype = int ), np.asarray( system.col, dtype = int ) )
    
    rows = np.append( E_data_quadratic[0].row, E_local_quadratic.row )
    cols = np.append( E_data_quadratic[0].col, E_local_quadratic.col )
    vals = np.append( E_data_quadratic[0].data, E_local_quadratic.data )
    system = cvxopt.spmatrix( vals, np.asarray( rows, dtype = int ), np.asarray( cols, dtype = int ) )
    
    # print( "solve_for_T() singular values:", np.linalg.svd( scipy.sparse.coo_matrix( ( vals, (rows, cols) ) ).todense(), compute_uv=False ) )
    
    rhs = cvxopt.matrix( -0.5*E_data_quadratic[1] )
    
    cvxopt.cholmod.linsolve( system, rhs )
    
    result = np.array( rhs ).squeeze()
    
    ## Reshape so that the first 12p elements become the first column, etc.
    T = result.reshape( ( 12*poses, -1 ), order = 'F' )
    
    assert len( T.shape ) == 2
    assert T.shape[0] % 12 == 0
    
    return T

def generateRandomData( poses = None, num_vertices = None ):
    np.random.seed(0)
    
    vs = np.random.random( ( num_vertices, 3 ) )
    vprimes = np.random.random( ( num_vertices, 3*poses ) )
    ## Some linearly changing data plus 0.1 * some noise.
    Ts = ( np.outer( np.linspace( 0, 1, num_vertices ), np.random.random( 12*poses ) ) + 0.1*np.random.random( ( num_vertices, 12*poses ) ) )
    
    def shuffled( a ):
        a = np.array(a)
        np.random.shuffle(a)
        return a
    
    neighbors = [ shuffled( list( set( np.arange( num_vertices ) ) - set([i]) ) )[:5] for i in range(num_vertices) ]
    
    return vs, vprimes, Ts, neighbors, poses, num_vertices

if __name__ == '__main__':
    np.set_printoptions( linewidth = 2000 )
    
    vs, vprimes, Ts, neighbors, poses, num_vertices = generateRandomData( poses = 1, num_vertices = 4 )
    assert len( Ts ) == num_vertices
    
    E_data = quadratic_for_E_data( vs, vprimes )
    
    for i in range( 1000 ):
        assert len( Ts ) == num_vertices
        
        ws_ssv_energy = [ solve_for_w( Ts[i], Ts[ neighbors[i] ].T, return_energy = True ) for i in range( num_vertices ) ]
        ws = [ w for w, ssv, energy in ws_ssv_energy ]
        print( "E_local from ws point of view:", np.sum([ energy for w, ssv, energy in ws_ssv_energy ]) )
        
        E_local = quadratic_for_E_local( neighbors, ws, poses )
        E_local_val = evaluate_E_local( E_local, Ts.T )
        print( "E_local from Ts point of view:", E_local_val )
        
        E_data_val = evaluate_E_data( E_data, Ts.T )
        print( "E_data from Ts point of view:", E_data_val )
        
        print( "=> E_total:", E_data_val + E_local_val )
        Ts = solve_for_T( E_data, E_local, poses ).T
        
        E_local_val = evaluate_E_local( E_local, Ts.T )
        print( "(E_local next from Ts point of view:", E_local_val, ")" )
        
        E_data_val = evaluate_E_data( E_data, Ts.T )
        print( "(E_data next from Ts point of view:", E_data_val, ")" )
        
        print( "Ts singular values:", np.linalg.svd( Ts, compute_uv = False ) )
    