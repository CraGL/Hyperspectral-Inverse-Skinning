"""
Sample code automatically generated on 2017-12-19 21:58:33

by www.matrixcalculus.org

from input

d/dB norm2(v*(p+B*-inv(B'*v'*v*B)*B'*v'*(v*p-w))-w) = -(1/norm2((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*((v*p-w)'*v*B*inv((v*B)'*v*B))-(1/norm2(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)*(((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v*B*inv((v*B)'*v*B))+1/norm2((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*((v*p-w)'*v*B*inv((v*B)'*v*B)))+1/norm2(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*v'*(v*p-w)*(((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v*B*inv((v*B)'*v*B)))
d/dp norm2(v*(p+B*-inv(B'*v'*v*B)*B'*v'*(v*p-w))-w) = 1/norm2((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)-1/norm2((p'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*(p-B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)

where

w is a vector
p is a vector
B is a matrix
v is a matrix

The generated code is provided"as is" without warranty of any kind.

The original expression can also be written as:
d/dp norm2((v*p-w)-v*B*inv(B'*v'*v*B)*B'*v'*(v*p-w)) = 1/norm2((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-1/norm2((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))
d/dB norm2((v*p-w)-v*B*inv(B'*v'*v*B)*B'*v'*(v*p-w)) = -(1/norm2((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*((v*p-w)'*v*B*inv((v*B)'*v*B))-(1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)*(((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v*B*inv((v*B)'*v*B))+1/norm2((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*((v*p-w)'*v*B*inv((v*B)'*v*B)))+1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*v'*(v*p-w)*(((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v*B*inv((v*B)'*v*B)))

second-derivative w.r.t p
d2/dp2 E = d/dp 1/norm2((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-1/norm2((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)) = 1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*v'*v-(1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)).^3*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*(((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v)-1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)).^3*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*(((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v))-1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*v'*v*B*inv((v*B)'*v*B)*B'*v'*v-(1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*v'*v*B*inv((v*B)'*v*B)*B'*v'*v-(1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)).^3*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*(((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v*B*inv((v*B)'*v*B)*B'*v'*v)-1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)).^3*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*(((v*p-w)'-(v*p-w)'*v*B*inv((v*B)'*v*B)*B'*v')*v*B*inv((v*B)'*v*B)*B'*v'*v))-1/norm2(v*p-w-v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w))*v'*v*B*inv((v*B)'*v*B)*B'*v'*v*B*inv((v*B)'*v*B)*B'*v'*v)
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
	assert(p_rows == v_cols == B_rows)
	assert(B_cols)
	assert(w_rows == v_rows)

	## 3p-by-handles = 3p-by-12p * 12p-by-handles
	vB = np.dot(v, B)

	## 3p-vector
	t_0 = (np.dot(v, p) - w)
	
	## handles-by-handles
	T_1 = np.linalg.inv(np.dot(vB.T, vB))
	## 12p-vector = 12p-by-3p * 3p-vector
	t_2 = np.dot(v.T, t_0)
	## 12p-vector = 12p-by-handles * handles-by-handles * handles-by-12p * 12p-vector
	t_3 = np.dot(B, np.dot(T_1, np.dot(B.T, t_2)))
	## 3p-vector
	t_4 = (np.dot(v, (p - t_3)) - w)
	## handles-vector = 3p-vector * 3p-by-handles * handles-by-handles
	t_5 = np.dot(np.dot(t_0, vB), T_1)
	## scalar
	t_6 = np.linalg.norm(t_4)
	## scalar
	if abs( t_6 ) > 1e-10:
		t_8 = (1 / t_6)
	else:
		t_8 = 1.
	
	## Why not?
	# t_8 = 1.0
	
	## 3p-vector
	# t_7 = (np.dot((p + np.dot(t_5, B.T)), v.T) - w)
	t_7 = t_4
	## 12p-vector = 12p-by-3p * 3p-vector
	t_9 = np.dot(v.T, t_4)
	## handles-vector = 3p-vector * 3p-by-handles * handles-by-handles
	t_11 = np.dot(np.dot(t_7, vB), T_1)
	functionValue = t_6
	
	# gradientB = (((t_8 * np.multiply.outer(t_9, t_5)) - ((t_8 * np.multiply.outer(np.dot(v.T, np.dot(v, t_3)), t_11)) + (t_8 * np.multiply.outer(np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_1, np.dot(B.T, t_9))))), t_5)))) + (t_8 * np.multiply.outer(t_2, t_11)))
	gradientB = -t_8 * (((np.multiply.outer(t_9, t_5)) - ((np.multiply.outer(np.dot(v.T, np.dot(v, t_3)), t_11)) + (np.multiply.outer(np.dot(v.T, np.dot(vB, np.dot(T_1, np.dot(B.T, t_9)))), t_5)))) + (np.multiply.outer(t_2, t_11)))

	tp_2 = t_4
	tp_4 = t_9
	# gradientp = ((tp_3 * tp_4) + (tp_3 * np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_1, np.dot(B.T, tp_4)))))))
	gradientp = t_8 * ((tp_4) - (np.dot(v.T, np.dot(vB, np.dot(T_1, np.dot(B.T, tp_4))))))

	return functionValue, gradientp, gradientB

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
	assert(p_rows == v_cols == B_rows)
	assert(B_cols)
	assert(w_rows == v_rows)

	t_0 = (np.dot(v, p) - w)
	T_1 = np.linalg.inv(np.dot(np.dot(np.dot(v, B).T, v), B))
	t_2 = np.dot(v.T, t_0)
	t_3 = np.dot(B, np.dot(T_1, np.dot(B.T, t_2)))
	t_4 = (np.dot(v, (p - t_3)) - w)
	t_5 = np.dot(np.dot(np.dot(t_0, v), B), T_1)
	t_6 = np.linalg.norm(t_4)
	t_7 = (np.dot((p - np.dot(t_5, B.T)), v.T) + -w)
	t_8 = (1 / np.linalg.norm(t_7))
	t_9 = np.dot(v.T, t_4)
	t_10 = (1 / t_6)
	t_11 = np.dot(np.dot(np.dot(t_7, v), B), T_1)
	functionValue = t_6
	gradient = -(((t_8 * np.multiply.outer(t_9, t_5)) - ((t_10 * np.multiply.outer(np.dot(v.T, np.dot(v, t_3)), t_11)) + (t_8 * np.multiply.outer(np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_1, np.dot(B.T, t_9))))), t_5)))) + (t_10 * np.multiply.outer(t_2, t_11)))

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
	assert(p_rows == v_cols == B_rows)
	assert(B_cols)
	assert(w_rows == v_rows)

	t_0 = (np.dot(v, p) - w)
	T_1 = np.linalg.inv(np.dot(np.dot(np.dot(v, B).T, v), B))
	t_2 = (np.dot(v, (p - np.dot(B, np.dot(T_1, np.dot(B.T, np.dot(v.T, t_0)))))) - w)
	t_3 = (1 / np.linalg.norm((np.dot((p - np.dot(np.dot(np.dot(np.dot(t_0, v), B), T_1), B.T)), v.T) + -w)))
	t_4 = np.dot(v.T, t_2)
	functionValue = np.linalg.norm(t_2)
	gradient = ((t_3 * t_4) - (t_3 * np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_1, np.dot(B.T, t_4)))))))

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

	t_0 = (np.dot(v, p) - w)
	T_1 = np.linalg.inv(np.dot(np.dot(np.dot(v, B).T, v), B))
	t_2 = (t_0 - np.dot(np.dot(np.dot(np.dot(np.dot(t_0, v), B), T_1), B.T), v.T))
	t_3 = (1 / np.linalg.norm(t_2))
	t_4 = (t_0 - np.dot(v, np.dot(B, np.dot(T_1, np.dot(B.T, np.dot(v.T, t_0))))))
	t_5 = np.dot(v.T, t_4)
	t_6 = np.linalg.norm(t_4)
	t_7 = (1 / (t_6 ** 3))
	t_8 = np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_1, np.dot(B.T, t_5)))))
	t_9 = np.dot(t_2, v)
	t_10 = (1 / t_6)
	T_11 = np.dot(v.T, v)
	T_12 = np.dot(np.dot(np.dot(np.dot(np.dot(T_11, B), T_1), B.T), v.T), v)
	T_13 = (t_10 * T_12)
	t_14 = np.dot(np.dot(np.dot(np.dot(np.dot(t_9, B), T_1), B.T), v.T), v)
	# functionValue = ((t_3 * t_5) - (t_3 * t_8))
	hessian = ((((t_10 * T_11) - ((t_7 * np.multiply.outer(t_5, t_9)) - (t_7 * np.multiply.outer(t_8, t_9)))) - T_13) - ((T_13 - ((t_7 * np.multiply.outer(t_5, t_14)) - (t_7 * np.multiply.outer(t_8, t_14)))) - (t_10 * np.dot(np.dot(np.dot(np.dot(np.dot(T_12, B), T_1), B.T), v.T), v))))

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
	P = 1
	handles = 2
	B = np.random.randn(12*P, handles)
	p = np.random.randn(12*P)
	v = np.random.randn(3*P, 12*P)
	w = np.random.randn(3*P)
	return B, p, v, w

if __name__ == '__main__':
	B, p, v, w = generateRandomData()
	functionValue, gradientp, gradientB = f_and_dfdp_and_dfdB(p, B, v, w)
	functionValue_dumb, gradientp_dumb, gradientB_dumb = f_and_dfdp_and_dfdB_dumb(p, B, v, w)
	
	print('functionValue = ', functionValue)
	print('gradient p = ', gradientp)
	print('gradient B = ', gradientB)
	
	print('functionValue_dumb = ', functionValue_dumb)
	print('gradient p dumb = ', gradientp_dumb)
	print('gradient B dumb = ', gradientB_dumb)
	
	print( "Function value matches if zero:", abs( functionValue - functionValue ) )
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
