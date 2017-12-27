"""
Sample code automatically generated on 2017-12-25 04:12:24

by www.matrixcalculus.org

from input

E = norm2((v*p-w)-v*(inv(I-A)*(I+A)*B)*inv((inv(I-A)*(I+A)*B)'*v'*v*(inv(I-A)*(I+A)*B))*(inv(I-A)*(I+A)*B)'*v'*(v*p-w))^2

where

v is vbar (3p-by-12p)
w is vprime (3p-vector)
p is a 12p-vector
I is the 12p identity matrix
B is the left (handles-1) columns of a 12p identity matrix (12p-by-(handles-1))

For the big inverse in the energy to work, we need:
    max(12p, handles, 3p) = max( 3p, handles ) >= handles
which is equivalent to:
    3p >= handles
(This wouldn't be a problem if we switched the big inverse to a pseudoinverse,
but that would make the gradient harder to calculate.)

d/dA norm2((v*p-w)-v*(inv(I-A)*(I+A)*B)*inv((inv(I-A)*(I+A)*B)'*v'*v*(inv(I-A)*(I+A)*B))*(inv(I-A)*(I+A)*B)'*v'*(v*p-w))^2 = -(2*inv(I-A)'*v'*(v*p-w-v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)'*v'*(v*p-w))*((v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)')+2*inv(I-A)'*v'*(v*p-w-v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)'*v'*(v*p-w))*((v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B')-(2*inv(I-A)'*v'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)'*v'*(v*p-w)*(((v*p-w)'-(v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)'*v')*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B')+2*inv(I-A)'*v'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)'*v'*(v*p-w)*(((v*p-w)'-(v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)'*v')*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)')+2*inv(I-A)'*v'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)'*v'*(v*p-w-v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)'*v'*(v*p-w))*((v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)')+2*inv(I-A)'*v'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)'*v'*(v*p-w-v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)'*v'*(v*p-w))*((v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'))+2*inv(I-A)'*v'*(v*p-w)*(((v*p-w)'-(v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)'*v')*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B')+2*inv(I-A)'*v'*(v*p-w)*(((v*p-w)'-(v*p-w)'*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(A+I)*B)'*v*inv(I-A)*(A+I)*B)*B'*(A+I)'*inv(I-A)'*v')*v*inv(I-A)*(A+I)*B*inv((v*inv(I-A)*(I+A)*B)'*v*inv(I-A)*(A+I)*B)*B'*(I+A)'*inv(I-A)'))

where

p is a vector
I is a matrix (symmetric matrix generates more not less code)
w is a vector
A is a matrix
B is a matrix
v is a matrix

The generated code is provided"as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

SKIP_CHECKS = False

def is_skew_symmetric( A, threshold = 1e-10 ):
    if SKIP_CHECKS: return True
    
    # return abs( A + A.T ).max() < threshold
    print( "A is skew symmetric if this is 0:", abs( A + A.T ).max() )
    return True

def is_orthogonal( Q, threshold = 1e-10 ):
    if SKIP_CHECKS: return True
    
    # return abs( np.dot( Q, Q.T ) - I ).max() < threshold
    print( "Q is orthogonal if this is 0:", abs( np.dot( Q.T, Q ) - np.eye(Q.shape[1]) ).max() )
    return True

## TODO: pack() and unpack() and A_from_non_Cayley_B() should use the Grassmann manifold parameters only
##       and zero the rest (or rotate appropriately).
def unpack( x, poses, handles ):
    p = x[:12*poses]
    
    xa = x[12*poses:]
    
    off = 0
    A = np.zeros( ( 12*poses, 12*poses ) )
    for row in range( 12*poses - 1 ):
        A[ row, row+1: ] = xa[ off : off + 12*poses-1-row ]
        off += 12*poses-1-row
    
    assert off == len( xa )
    
    A -= A.T
    
    ## A should be skew-symmetric
    assert is_skew_symmetric( A )
    
    return p, A

def pack( p, A, poses, handles ):
    ## A should be skew-symmetric
    assert is_skew_symmetric( A )
    
    assert len(p) % 12 == 0
    assert poses == len(p)//12
    # x[:12*poses] = p
    
    xa = np.zeros( (6*poses)*(12*poses-1) )
    
    off = 0
    for row in range( 12*poses - 1 ):
        xa[ off : off + 12*poses-1-row ] = A[ row, row+1: ]
        off += 12*poses-1-row
    
    assert off == len( xa )
    
    x = np.concatenate( ( p.squeeze(), xa ) )
    
    return x

def B_from_Cayley_A( A, handles ):
    ## A should be skew-symmetric
    assert is_skew_symmetric( A )
    
    I = np.eye(A.shape[0])
    ## Return: Q = (I-A)^(-1) * (I+A)
    # Q = np.linalg.solve( I-A, I+A )[:B_rows]
    Q = np.dot( np.linalg.inv( I-A ), I+A )[:,:handles]
    assert is_orthogonal( Q )
    return Q

def A_from_non_Cayley_B( Q ):
    ## This function follows the paper mentioned below. Its input is Q and output is X.
    handles = Q.shape[1]
    
    ## The Representation and Parametrization of Orthogonal Matrices (Ron Shepard, Scott R. Brozell, Gergely Gidofalvi 2015 Journal of Physical Chemistry)
    ## Equations 101-103:
    Q1 = Q[:handles,:handles]
    ## TODO: For Grassmann parameters, use SVD of Q1 to get a right-matrix to modify B.
    Q2 = Q[handles:,:handles]
    I = np.eye(handles)
    F = np.dot( (I-Q1), np.linalg.inv( I+Q1 ) )
    B = 0.5*( F.T - F )
    A = 0.5*np.dot( Q2, ( I + F ) )
    X = np.zeros( ( Q.shape[0], Q.shape[0] ) )
    X[:handles,:handles] = B
    X[handles:,:handles] = A
    X[:handles,handles:] = -A.T
    return X

def A_from_non_Cayley_B_broken( B ):
    # raise RuntimeError( "This function is broken." )
    
    handles = B.shape[1]
    
    ## Complete B. Get an orthonormal basis for the whole space.
    _, S, V = np.linalg.svd( B.T )
    assert is_orthogonal( V )
    ## The first handles rows of V are the ones that span the columns of B.
    ## We want the Cayley transform for V.T
    I = np.eye( B.shape[0] )
    # A = np.linalg.solve( I+V, I-V )
    ## Why does (I+V) or (I+V.T) have a zero eigenvalue/singular value?
    A = np.dot( np.linalg.inv( I+V.T ), I-V.T )
    B_recovered = B_from_Cayley_A( A, B.shape[0] ) # handles )
    ## This would be non-zero, since B_recovered should be orthogonal:
    # print( 'A_from_non_Cayley_B() recovery difference:', abs( B - B_recovered ).max() )
    print( 'A_from_non_Cayley_B() recovery difference:', abs( B_recovered[:handles].dot( B ) ).max() )
    ## A should be skew-symmetric
    assert is_skew_symmetric( A )
    return A

def f_and_dfdp_and_dfdA_matrixcalculus(p, A, v, w, handles):
    I = np.eye(len(p))
    ## B is the matrix which takes the top (handles-1) rows.
    ## It's a truncated identity matrix.
    B = I.copy()[:,:handles]
    
    assert(type(A) == np.ndarray)
    dim = A.shape
    assert(len(dim) == 2)
    A_rows = dim[0]
    A_cols = dim[1]
    
    ## A should be skew-symmetric
    assert is_skew_symmetric( A )
    
    assert(type(B) == np.ndarray)
    dim = B.shape
    assert(len(dim) == 2)
    B_rows = dim[0]
    B_cols = dim[1]
    assert(type(I) == np.ndarray)
    dim = I.shape
    assert(len(dim) == 2)
    I_rows = dim[0]
    I_cols = dim[1]
    assert(type(p) == np.ndarray)
    dim = p.shape
    assert(len(dim) == 1)
    p_rows = dim[0]
    assert(type(v) == np.ndarray)
    dim = v.shape
    assert(len(dim) == 2)
    v_rows = dim[0]
    v_cols = dim[1]
    assert(type(w) == np.ndarray)
    dim = w.shape
    assert(len(dim) == 1)
    w_rows = dim[0]
    assert(A_cols == v_cols == p_rows == I_cols)
    assert(I_rows == A_rows)
    assert(A_cols == v_cols == B_rows == I_cols)
    assert(I_cols == A_cols == B_rows == p_rows == v_cols)
    assert(B_cols)
    assert(v_rows == w_rows)

    T_0 = np.linalg.inv((I - A))
    T_1 = (A + I)
    # T_2 = (I + A)
    T_2 = T_1
    t_3 = (np.dot(v, p) - w)
    T_4 = np.linalg.inv(np.dot(np.dot(np.dot(np.dot(np.dot(v, np.dot(T_0, np.dot(T_2, B))).T, v), T_0), T_1), B))
    t_5 = np.dot(T_0.T, np.dot(v.T, t_3))
    t_6 = np.dot(v, np.dot(T_0, np.dot(T_1, np.dot(B, np.dot(T_4, np.dot(B.T, np.dot(T_2.T, t_5)))))))
    t_7 = (t_3 - t_6)
    t_8 = np.dot(T_0.T, np.dot(v.T, t_7))
    T_9 = np.linalg.inv(np.dot(np.dot(np.dot(np.dot(np.dot(v, np.dot(T_0, np.dot(T_1, B))).T, v), T_0), T_1), B))
    t_10 = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(t_3, v), T_0), T_1), B), T_9), B.T)
    t_11 = np.dot(np.dot(t_10, T_1.T), T_0.T)
    t_12 = np.dot(T_0.T, np.dot(v.T, t_6))
    t_13 = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot((t_3 - np.dot(t_11, v.T)), v), T_0), T_1), B), T_4), B.T)
    extra = np.dot(v.T, np.dot(v, np.dot(T_0, np.dot(T_1, np.dot(B, np.dot(T_9, np.dot(B.T, np.dot(T_1.T, t_8))))))))
    t_14 = np.dot(T_0.T, extra)
    t_15 = np.dot(np.dot(t_13, T_2.T), T_0.T)
    functionValue = (np.linalg.norm(t_7) ** 2)
    gradientA = -(((((2 * np.multiply.outer(t_8, t_11)) + (2 * np.multiply.outer(t_8, t_10))) - ((((2 * np.multiply.outer(t_12, t_13)) + (2 * np.multiply.outer(t_12, t_15))) + (2 * np.multiply.outer(t_14, t_11))) + (2 * np.multiply.outer(t_14, t_10)))) + (2 * np.multiply.outer(t_5, t_13))) + (2 * np.multiply.outer(t_5, t_15)))

    # print( 'inner B:', B.shape )
    # print( np.dot(np.dot(T_0, T_1), B) )
    
    t_5 = np.dot(v.T, t_7)
    gradientp = ((2 * t_5) - (2 * extra))

    return functionValue, gradientp, gradientA

def f_and_dfdp_and_dfdA_matrixcalculus2(p, A, v, w, handles):
    I = np.eye(len(p))
    ## B is the matrix which takes the top (handles-1) rows.
    ## It's a truncated identity matrix.
    B = I.copy()[:,:handles]
    
    assert(type(A) == np.ndarray)
    dim = A.shape
    assert(len(dim) == 2)
    A_rows = dim[0]
    A_cols = dim[1]
    
    ## A should be skew-symmetric
    assert is_skew_symmetric( A )
    
    assert(type(B) == np.ndarray)
    dim = B.shape
    assert(len(dim) == 2)
    B_rows = dim[0]
    B_cols = dim[1]
    assert(type(I) == np.ndarray)
    dim = I.shape
    assert(len(dim) == 2)
    I_rows = dim[0]
    I_cols = dim[1]
    assert(type(p) == np.ndarray)
    dim = p.shape
    assert(len(dim) == 1)
    p_rows = dim[0]
    assert(type(v) == np.ndarray)
    dim = v.shape
    assert(len(dim) == 2)
    v_rows = dim[0]
    v_cols = dim[1]
    assert(type(w) == np.ndarray)
    dim = w.shape
    assert(len(dim) == 1)
    w_rows = dim[0]
    assert(A_cols == v_cols == p_rows == I_cols)
    assert(I_rows == A_rows)
    assert(A_cols == v_cols == B_rows == I_cols)
    assert(I_cols == A_cols == B_rows == p_rows == v_cols)
    assert(B_cols)
    assert(v_rows == w_rows)
    
    assert len(p)%12 == 0
    poses = len(p)//12
    
    T_0 = np.linalg.inv((I - A))
    T_1 = (A + I)
    T_01 = np.dot(T_0, T_1)
    
    vQ = np.dot(v, np.dot(T_01, B))
    
    t_3 = (np.dot(v, p) - w)
    T_4 = np.linalg.inv(np.dot(vQ.T, vQ))
    t_5 = np.dot(T_0.T, np.dot(v.T, t_3))
    t_6 = np.dot(vQ, np.dot(T_4, np.dot( vQ.T, t_3)))
    t_7 = (t_3 - t_6)
    t_8 = np.dot(T_0.T, np.dot(v.T, t_7))
    T_9 = T_4
    t_10 = np.dot(np.dot(np.dot(t_3, vQ), T_9), B.T)
    t_11 = np.dot(t_10, T_01.T)
    t_12 = np.dot(T_0.T, np.dot(v.T, t_6))
    t_13 = np.dot(np.dot(np.dot((t_3 - np.dot(t_11, v.T)), vQ), T_4), B.T)
    extra = np.dot(v.T, np.dot(v, np.dot(T_01, np.dot(B, np.dot(T_9, np.dot(B.T, np.dot(T_1.T, t_8)))))))
    t_14 = np.dot(T_0.T, extra)
    t_15 = np.dot(t_13, T_01.T)
    functionValue = (np.linalg.norm(t_7) ** 2)
    gradientA = -(((((2 * np.multiply.outer(t_8, t_11)) + (2 * np.multiply.outer(t_8, t_10))) - ((((2 * np.multiply.outer(t_12, t_13)) + (2 * np.multiply.outer(t_12, t_15))) + (2 * np.multiply.outer(t_14, t_11))) + (2 * np.multiply.outer(t_14, t_10)))) + (2 * np.multiply.outer(t_5, t_13))) + (2 * np.multiply.outer(t_5, t_15)))

    # print( 'inner B:', B.shape )
    # print( np.dot(np.dot(T_0, T_1), B) )
    
    t_5 = np.dot(v.T, t_7)
    gradientp = ((2 * t_5) - (2 * extra))

    return functionValue, gradientp, gradientA

f_and_dfdp_and_dfdA_matrixcalculus = f_and_dfdp_and_dfdA_matrixcalculus2

def f_and_dfdp_and_Hfp(p, A, v, w, handles):
    B = B_from_Cayley_A( A, handles )
    # print( 'B_from_Cayley_A:', B.shape )
    # print( B )
    
    assert(type(B) == np.ndarray)
    dim = B.shape
    assert(len(dim) == 2)
    B_rows = dim[0]
    B_cols = dim[1]
    assert(type(p) == np.ndarray)
    dim = p.shape
    assert(len(dim) == 1)
    p_rows = dim[0]
    assert(type(v) == np.ndarray)
    dim = v.shape
    assert(len(dim) == 2)
    v_rows = dim[0]
    v_cols = dim[1]
    assert(type(w) == np.ndarray)
    dim = w.shape
    assert(len(dim) == 1)
    w_rows = dim[0]
    assert(p_rows == v_cols == B_rows)
    assert(B_cols)
    assert(w_rows == v_rows)
    
    vB = np.dot( v, B )
    A = np.dot(np.dot(vB, np.linalg.inv(np.dot(vB.T,vB)) ), vB.T)
    foo = np.eye( A.shape[0] ) - A
    S = np.dot( foo, v )
    r = np.dot( foo, w )
    Q = np.dot( S.T, S )
    L = np.dot( S.T, r )
    C = np.dot( r.T, r )
    
    functionValue = np.dot( p.T, np.dot( Q, p ) ) - 2*np.dot( p.T, L ) + C
    gradient = 2 * ( np.dot( Q, p ) - L )
    hessian = 2 * Q
    
    return functionValue, gradient, hessian

def f_and_dfdp_and_dfdA( p, A, v, w, handles ):
    f, gradp, gradA = f_and_dfdp_and_dfdA_matrixcalculus( p, A, v, w, handles )
    
    # f2, gradp2, gradA2 = f_and_dfdp_and_dfdA_matrixcalculus2( p, A, v, w, handles )
    # print( "Faster? zero?" )
    # print( abs( f - f2 ) )
    # print( abs( gradp - gradp2 ).max() )
    # print( abs( gradA - gradA2 ).max() )
    
    ## gradient p check (computed another way):
    ## This test passes.
    # f, gradp2, hessp = f_and_dfdp_and_Hfp( p, A, v, w, handles )
    # print( '|gradient p difference| max:', abs( gradp - gradp2 ).max() )
    
    return f, gradp, gradA

def random_skew_symmetric_matrix( n ):
    A = np.random.randn(n,n)
    return 0.5 * ( A - A.T )

def generateRandomData():
    # np.random.seed(0)
    P = 1
    handles = 2
    
    ## If this isn't true, the inv() in the energy will fail.
    assert 3*P >= handles
    
    # B = np.random.randn(12*P, handles)
    # A = A_from_non_Cayley_B( B )
    A = random_skew_symmetric_matrix( 12*P )
    assert is_skew_symmetric( A )
    p = np.random.randn(12*P)
    v = np.random.randn(3*P, 12*P)
    w = np.random.randn(3*P)
    return p, A, v, w, P, handles

if __name__ == '__main__':
    p, A, v, w, poses, handles = generateRandomData()
    
    f, gradp, gradA = f_and_dfdp_and_dfdA( p, A, v, w, handles )
    f2, gradp2, hessp = f_and_dfdp_and_Hfp( p, A, v, w, handles )
    
    print( 'function value:', f )
    print( 'other function value:', f2 )
    print( '|function difference|:', abs( f - f2 ) )
    print( 'gradient p:', gradp )
    print( 'other gradient p:', gradp )
    print( '|gradient p difference| max:', abs( gradp - gradp2 ).max() )
    print( 'gradient A:', gradA )
    
    x = pack( p, A, poses, handles )
    p2, A2 = unpack( x, poses, handles )
    x2 = pack( p2, A2, poses, handles )
    print( "If pack/unpack work, these should be zeros:" )
    print( abs( p - p2 ).max() )
    print( abs( A - A2 ).max() )
    print( abs( x - x2 ).max() )
    
    def f_gradf_packed( x ):
        xp, xA = unpack( x, poses, handles )
        val, gradp, gradA = f_and_dfdp_and_dfdA( xp, xA, v, w, handles )
        grad = pack( gradp, gradA, poses, handles )
        return val, grad
    import scipy.optimize
    grad_err = scipy.optimize.check_grad( lambda x: f_gradf_packed(x)[0], lambda x: f_gradf_packed(x)[1], pack( p, A, poses, handles ) )
    print( "scipy.optimize.check_grad() error:", grad_err )
