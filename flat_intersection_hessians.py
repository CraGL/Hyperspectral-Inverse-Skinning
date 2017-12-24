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
	
import autograd.numpy as np
import autograd

from flat_intersection import pack, unpack

def f( x, vbar, vprime, poses ):
	v = vbar
	w = vprime
	
	p, B = unpack( x, poses )
	p = p.squeeze()
	
	# assert(type(B) == np.ndarray)
	dim = B.shape
	assert(len(dim) == 2)
	B_rows = dim[0]
	B_cols = dim[1]
	# assert(type(p) == np.ndarray)
	dim = p.shape
	assert(len(dim) == 1)
	p_rows = dim[0]
	# assert(type(v) == np.ndarray)
	dim = v.shape
	assert(len(dim) == 2)
	v_rows = dim[0]
	v_cols = dim[1]
	# assert(type(w) == np.ndarray)
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
	
	return functionValue

hess = autograd.hessian( f, 0 )

def fAndGpAndHp_fast(p, B, vbar, vprime):
	v = vbar
	w = vprime
	
	## Can't assume np.ndarray because of autograd.
	# assert(type(B) == np.ndarray)
	dim = B.shape
	assert(len(dim) == 2)
	B_rows = dim[0]
	B_cols = dim[1]
	# assert(type(p) == np.ndarray)
	dim = p.shape
	assert(len(dim) == 1)
	p_rows = dim[0]
	# assert(type(v) == np.ndarray)
	dim = v.shape
	assert(len(dim) == 2)
	v_rows = dim[0]
	v_cols = dim[1]
	# assert(type(w) == np.ndarray)
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

def dfdp( xp, xb, vbar, vprime, poses ):
	## Adapted from flat_intersection.unpack()
	p = xp.squeeze()
	B = xb.reshape(-1,12*poses).T
	
	return fAndGpAndHp_fast( p, B, vbar, vprime )[1]
def dfdp2( xp, xb, vbar, vprime, poses ):
	## Adapted from flat_intersection.unpack()
	p = xp.squeeze()
	B = xb.reshape(-1,12*poses).T
	
	return fAndGpAndHp_fast( p, B, vbar, vprime )[2]

def dfdB( xp, xb, vbar, vprime, poses):
	v = vbar
	w = vprime
	
	p = xp.squeeze()
	## Adapted from flat_intersection.unpack()
	B = xb.reshape(-1,12*poses).T
	
	# assert(type(B) == np.ndarray)
	dim = B.shape
	assert(len(dim) == 2)
	B_rows = dim[0]
	B_cols = dim[1]
	# assert(type(p) == np.ndarray)
	dim = p.shape
	assert(len(dim) == 1)
	p_rows = dim[0]
	# assert(type(v) == np.ndarray)
	dim = v.shape
	assert(len(dim) == 2)
	v_rows = dim[0]
	v_cols = dim[1]
	# assert(type(w) == np.ndarray)
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
	#gradientB = -(((2 * np.outer(t_6, t_5)) - ((2 * np.outer(np.dot(v.T, np.dot(v, t_3)), t_7)) + (2 * np.outer(np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_0, np.dot(B.T, t_6))))), t_5)))) + (2 * np.outer(t_2, t_7)))
	gradientB = -2 * (((np.outer(t_6, t_5)) - ((np.outer(np.dot(v.T, np.dot(v, t_3)), t_7)) + (np.outer(np.dot(v.T, np.dot(vB, np.dot(T_0, np.dot(B.T, t_6)))), t_5)))) + (np.outer(t_2, t_7)))
	## This matches pack()
	return gradientB.T.ravel()

dfdB2 = autograd.jacobian( dfdB, 1 )
# dfdpdB = autograd.jacobian( dfdB, 0 )
dfdpdB = autograd.jacobian( dfdp, 1 )

def hess_fast( x, vbar, vprime, poses ):
	NP = 12*poses
	
	xp = x[:NP]
	xb = x[NP:]
	
	result = np.zeros( ( len(x), len(x) ) )
	result[ :NP, :NP ] = dfdp2( xp, xb, vbar, vprime, poses )
	result[ :NP, NP: ] = dfdpdB( xp, xb, vbar, vprime, poses )
	result[ NP:, NP: ] = dfdB2( xp, xb, vbar, vprime, poses )
	## dpdB is the transpose of dBdp
	result[ NP:, :NP ] = result[ :NP, NP: ].T
	
	return result

hess = hess_fast

def generateRandomData(P):
	#np.random.seed(0)
	handles = 2
	B = np.random.randn(12*P, handles)
	p = np.random.randn(12*P)
	v = np.random.randn(3*P, 12*P)
	w = np.random.randn(3*P)
	return B, p, v, w

if __name__ == '__main__':
	poses = 1
	B, p, v, w = generateRandomData(poses)
	x = pack( p, B )
	
	import flat_intersection_direct_gradients
	
	functionValue = f( x, v, w, poses )
	print('functionValue = ', functionValue)
	hessValue = hess( x, v, w, poses )
	print( 'hessian shape:', hessValue.shape )
	
	f_fast, gp_fast, hp_fast = flat_intersection_direct_gradients.fAndGpAndHp_fast( p, B, v, w )
	
	print('functionValue_fast = ', f_fast)
	
	hp_auto = hessValue[:12*poses,:12*poses]
	
	print('hess p fast = ', hp_fast )
	print('hess p auto = ', hp_auto )
	
	print( "Function value matches if zero:", abs( functionValue - f_fast ) )
	print( "hess p matches if zero:", abs( hp_auto - hp_fast ).max() )
	
	hess_fast = hess_fast( x, v, w, poses )
	print('hess fast matches hess if zero:', abs( hess_fast - hessValue ).max() )
	