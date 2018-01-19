"""
Sample code automatically generated on 2017-12-23 07:16:38

by www.matrixcalculus.org

from input

d/dB norm2(v*(p+B*-inv(B'*v'*v*B)*B'*v'*(v*p-w))-w)^2 = -(2*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*((v*p-w)'*v*B*inv((v*B)'*v*B))-(2*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)*(((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v*B*inv((v*B)'*v*B))+2*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*((v*p-w)'*v*B*inv((v*B)'*v*B)))+2*v'*(v*p-w)*(((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v*B*inv((v*B)'*v*B)))
d/dp norm2(v*(p+B*-inv(B'*v'*v*B)*B'*v'*(v*p-w))-w)^2 = 2*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)-2*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)

where

w is a vector
p is a vector
B is a matrix
v is a matrix

The generated code is provided"as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import
	
import numpy as np
 
def f_and_dfdp_and_dfdB(p, B, vbar, vprime):
	v = vbar
	w = vprime
	
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
	assert(B_rows == p_rows == v_cols)
	assert(B_cols)
	assert(v_rows == w_rows)

	## 3p-by-handles = 3p-by-12p * 12p-by-handles
	vB = np.dot(v, B)

	## handles-by-handles
	T_0 = np.linalg.inv(np.dot(vB.T, vB))
	## 3p-vector
	t_1 = (np.dot(v, p) - w)
	## 12p-vector
	t_2 = np.dot(v.T, t_1)
	## 12p-vector = 12p-by-handles * handles-by-handles * handles-by-12p * 12p-vector 
	t_3 = np.dot(B, np.dot(T_0, np.dot(B.T, t_2)))
	## 3p-vector
	t_4 = (np.dot(v, (p - t_3)) - w)
	## handles-vector = 3p-vector * 3p-by-handles * handles-by-handles
	t_5 = np.dot(np.dot(t_1, vB), T_0)
	## 12p-vector
	t_6 = np.dot(v.T, t_4)
	## 12p-by-handles
	t_7 = np.dot(np.dot((np.dot((p - np.dot(t_5, B.T)), v.T) + -w), vB), T_0)
	functionValue = (np.linalg.norm(t_4) ** 2)
	#gradientB = -(((2 * np.multiply.outer(t_6, t_5)) - ((2 * np.multiply.outer(np.dot(v.T, np.dot(v, t_3)), t_7)) + (2 * np.multiply.outer(np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_0, np.dot(B.T, t_6))))), t_5)))) + (2 * np.multiply.outer(t_2, t_7)))
	gradientB = -2 * (((np.multiply.outer(t_6, t_5)) - ((np.multiply.outer(np.dot(v.T, np.dot(v, t_3)), t_7)) + (np.multiply.outer(np.dot(v.T, np.dot(vB, np.dot(T_0, np.dot(B.T, t_6)))), t_5)))) + (np.multiply.outer(t_2, t_7)))
	gradientp = 2 * (t_2 - np.dot(v.T, np.dot(vB, np.dot(T_0, np.dot(B.T, t_2)))))
	
	return functionValue, gradientp, gradientB 

def f_and_dfdp_and_dfdB_hand(p, B, vbar, vprime):
	v = vbar
	w = vprime
	
	## Speed this up! v is block diagonal.
	# vB = np.dot( v, B )
	vB = np.dot( v[0,:4], B.T.reshape( -1, 4 ).T ).reshape( B.shape[1], -1 ).T
	# vp = np.dot( v,p )
	vp = np.dot( v[0,:4], p.reshape( -1, 4 ).T ).ravel()
	
	S = np.dot( vB.T, vB )
	u = ( vp - w ).reshape(-1,1)
	R = np.dot( vB, np.linalg.inv(S) )
	Q = np.dot( R, vB.T )
	M = u - np.dot( Q, u )
	# MuR = np.dot( np.dot( M, u.T ), R )
	## Actually, M'*R is identically zero.
	# uMR = np.dot( np.dot( u, M.T ), R )
	assert len( u.shape ) == 2
	assert len( M.shape ) == 2
	
	E = ( M * M ).sum()
	
	# dE/dp = 2*v'*M
	gradp = 2 * np.dot( v.T, M )
	
	# dE/dB = - dE/dp * (u'*R)
	gradB = np.dot( -gradp, np.dot( u.T, R ) )
	
	return E, gradp.squeeze(), gradB

def fAndGB(p, B, vbar, vprime):
	v = vbar
	w = vprime
	
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
	assert(B_rows == p_rows == v_cols)
	assert(B_cols)
	assert(v_rows == w_rows)

	T_0 = np.linalg.inv(np.dot(np.dot(np.dot(v, B).T, v), B))
	t_1 = (np.dot(v, p) - w)
	t_2 = np.dot(v.T, t_1)
	t_3 = np.dot(B, np.dot(T_0, np.dot(B.T, t_2)))
	t_4 = (np.dot(v, (p - t_3)) - w)
	t_5 = np.dot(np.dot(np.dot(t_1, v), B), T_0)
	t_6 = np.dot(v.T, t_4)
	t_7 = np.dot(np.dot(np.dot((np.dot((p - np.dot(t_5, B.T)), v.T) + -w), v), B), T_0)
	functionValue = (np.linalg.norm(t_4) ** 2)
	gradient = -(((2 * np.multiply.outer(t_6, t_5)) - ((2 * np.multiply.outer(np.dot(v.T, np.dot(v, t_3)), t_7)) + (2 * np.multiply.outer(np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_0, np.dot(B.T, t_6))))), t_5)))) + (2 * np.multiply.outer(t_2, t_7)))

	return functionValue, gradient

def fAndGp(p, B, vbar, vprime):
	v = vbar
	w = vprime
	
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
	assert(B_rows == p_rows == v_cols)
	assert(B_cols)
	assert(v_rows == w_rows)

	T_0 = np.linalg.inv(np.dot(np.dot(np.dot(v, B).T, v), B))
	t_1 = (np.dot(v, (p - np.dot(B, np.dot(T_0, np.dot(B.T, np.dot(v.T, (np.dot(v, p) - w))))))) - w)
	t_2 = np.dot(v.T, t_1)
	functionValue = (np.linalg.norm(t_1) ** 2)
	gradient = ((2 * t_2) - (2 * np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_0, np.dot(B.T, t_2)))))))

	return functionValue, gradient
	
def f_and_dfdp_and_dfdB_dumb( p, B, vbar, vprime ):
	f, dp = fAndGp( p, B, vbar, vprime )
	f2, dB = fAndGB( p, B, vbar, vprime )
	
	assert abs( f - f2 ) < 1e-10
	
	return f, dp, dB
	
def d2f_dp2_dumb( p, B, vbar, vprime ):
	v = vbar
	w = vprime

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
	assert(B_rows == p_rows == v_cols)
	assert(B_cols)
	assert(v_rows == w_rows)

	T_0 = np.linalg.inv(np.dot(np.dot(np.dot(v, B).T, v), B))
	t_1 = np.dot(v.T, (np.dot(v, (p - np.dot(B, np.dot(T_0, np.dot(B.T, np.dot(v.T, (np.dot(v, p) - w))))))) - w))
	T_2 = np.dot(v.T, v)
	T_3 = np.dot(np.dot(np.dot(np.dot(np.dot(T_2, B), T_0), B.T), v.T), v)
	T_4 = (2 * T_3)
	#functionValue = ((2 * t_1) - (2 * np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_0, np.dot(B.T, t_1)))))))
	hessian = (((2 * T_2) - T_4) - (T_4 - (2 * np.dot(np.dot(np.dot(np.dot(np.dot(T_3, B), T_0), B.T), v.T), v))))

	return hessian
	
def fAndGpAndHp_fast(p, B, vbar, vprime):
	v = vbar
	w = vprime
	
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

def generateRandomData():
	#np.random.seed(0)
	P = 2
	handles = 3
	## If this isn't true, the inv() in the energy will fail.
	assert 3*P >= handles
	B = np.random.randn(12*P, handles)
	p = np.random.randn(12*P)
	v = np.kron( np.eye( 3*P ), np.append( np.random.randn(3), [1.] ) )
	w = np.random.randn(3*P)
	return B, p, v, w, P, handles

def pack( point, B ):
	'''
	`point` is a 12P-by-1 column matrix.
	`B` is a 12P-by-#(handles-1) matrix.
	Returns them packed so that unpack( pack( point, B ) ) == point, B.
	'''
	p12 = B.shape[0]
	handles = B.shape[1]
	X = np.zeros( p12*(handles+1) )
	X[:p12] = point.ravel()
	X[p12:] = B.T.ravel()
	return X

def unpack( X, poses ):
	'''
	X is a flattened array with #handle*12P entries.
	The first 12*P entries are `point` as a 12*P-by-1 matrix.
	The remaining entries are the 12P-by-#(handles-1) matrix B.
	
	where P = poses.
	'''
	P = poses
	point = X[:12*P].reshape(12*P, 1)
	B = X[12*P:].reshape(-1,12*P).T

	return point, B

if __name__ == '__main__':
	B, p, v, w, poses, handles = generateRandomData()
	functionValue, gradientp, gradientB = f_and_dfdp_and_dfdB(p, B, v, w)
	functionValue_dumb, gradientp_dumb, gradientB_dumb = f_and_dfdp_and_dfdB_dumb(p, B, v, w)
	
	print('functionValue = ', functionValue)
	print('gradient p = ', gradientp)
	print('gradient B = ', gradientB)
	
	print('functionValue_dumb = ', functionValue_dumb)
	print('gradient p dumb = ', gradientp_dumb)
	print('gradient B dumb = ', gradientB_dumb)
	
	print( "Function value matches if zero:", abs( functionValue - functionValue_dumb ) )
	print( "gradient p matches if zero:", abs( gradientp - gradientp_dumb ).max() )
	print( "gradient B matches if zero:", abs( gradientB - gradientB_dumb ).max() )
	
	f_fast, gp_fast, hp_fast = fAndGpAndHp_fast( p, B, v, w )
	hp_dumb = d2f_dp2_dumb( p, B, v, w )
	
	print('functionValue_fast = ', f_fast)
	print('gradient p fast = ', gp_fast )
	print('hess p fast = ', hp_fast )
	print('hess p dumb = ', hp_dumb )
	
	print( "Function value matches if zero:", abs( functionValue - f_fast ) )
	print( "gradient p matches if zero:", abs( gradientp - gp_fast ).max() )
	print( "hess p matches if zero:", abs( hp_dumb - hp_fast ).max() )
	
	f_hand, gradp_hand, gradB_hand = f_and_dfdp_and_dfdB_hand(p, B, v, w)
	print( 'f hand = ', f_hand )
	print( 'gradient p hand = ', gradp_hand )
	print( 'gradient B hand = ', gradB_hand )
	print( "Function value matches if zero:", abs( functionValue - f_hand ) )
	print( "gradient p matches if zero:", abs( gradientp - gradp_hand ).max() )
	print( "gradient B matches if zero:", abs( gradientB - gradB_hand ).max() )
	
	def f_gradf_packed( x ):
		xp, xB = unpack( x, poses )
		xp = xp.squeeze()
		val, gradp, gradB = f_and_dfdp_and_dfdB( xp, xB, v, w )
		grad = pack( gradp, gradB )
		return val, grad
	import scipy.optimize
	grad_err = scipy.optimize.check_grad( lambda x: f_gradf_packed(x)[0], lambda x: f_gradf_packed(x)[1], pack( p, B ) )
	print( "scipy.optimize.check_grad() error:", grad_err )

## f_and_dfdp_and_dfdB_hand() wins
f_and_dfdp_and_dfdB = f_and_dfdp_and_dfdB_hand
