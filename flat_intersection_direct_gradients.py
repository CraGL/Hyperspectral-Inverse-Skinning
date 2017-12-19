"""
Sample code automatically generated on 2017-12-19 21:58:33

by www.matrixcalculus.org

from input

d/dB norm2(v*(p+B*inv(B'*v'*v*B)*B'*v'*(v*p-w))-w) = 1/norm2((p'+(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v'*(v*(p+B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*((v*p-w)'*v*B*inv((v*B)'*v*B))-(1/norm2(v*(p+B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*p-w)*(((p'+(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v*B*inv((v*B)'*v*B))+1/norm2((p'+(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v'*v*B*inv((v*B)'*v*B)*B'*v'*(v*(p+B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*((v*p-w)'*v*B*inv((v*B)'*v*B)))+1/norm2(v*(p+B*inv((v*B)'*v*B)*B'*v'*(v*p-w))-w)*v'*(v*p-w)*(((p'+(v*p-w)'*v*B*inv((v*B)'*v*B)*B')*v'+(-w)')*v*B*inv((v*B)'*v*B))

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
	t_4 = (np.dot(v, (p + t_3)) - w)
	## handles-vector = 3p-vector * 3p-by-handles * handles-by-handles
	t_5 = np.dot(np.dot(t_0, vB), T_1)
	## scalar
	t_6 = np.linalg.norm(t_4)
	## scalar
	if abs( t_6 ) > 1e-6:
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
	gradientB = t_8 * (((np.multiply.outer(t_9, t_5)) - ((np.multiply.outer(np.dot(v.T, np.dot(v, t_3)), t_11)) + (np.multiply.outer(np.dot(v.T, np.dot(vB, np.dot(T_1, np.dot(B.T, t_9)))), t_5)))) + (np.multiply.outer(t_2, t_11)))

	tp_2 = t_4
	tp_4 = t_9
	# gradientp = ((tp_3 * tp_4) + (tp_3 * np.dot(v.T, np.dot(v, np.dot(B, np.dot(T_1, np.dot(B.T, tp_4)))))))
	gradientp = t_8 * ((tp_4) + (np.dot(v.T, np.dot(vB, np.dot(T_1, np.dot(B.T, tp_4))))))

	return functionValue, gradientp, gradientB



def generateRandomData():
	np.random.seed(0)
	B = np.random.randn(3, 3)
	p = np.random.randn(3)
	v = np.random.randn(3, 3)
	w = np.random.randn(3)
	return B, p, v, w

if __name__ == '__main__':
	B, p, v, w = generateRandomData()
	functionValue, gradientp, gradientB = f_and_dfdp_and_dfdB(B, p, v, w)
	print('functionValue = ', functionValue)
	print('gradient p = ', gradientp)
	print('gradient B = ', gradientB)

