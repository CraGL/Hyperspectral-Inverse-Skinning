"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division

import os
import sys
import argparse
import time
import numpy as np
from numpy.linalg import svd
import scipy.optimize

import format_loader
from trimesh import TriMesh

MAX_H = 20

## http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
def nullspace(A, atol=1e-13, rtol=0):
	"""Compute an approximate basis for the nullspace of A.

	The algorithm used by this function is based on the singular value
	decomposition of `A`.

	Parameters
	----------
	A : ndarray
		A should be at most 2-D.  A 1-D array with length k will be treated
		as a 2-D with shape (1, k)
	atol : float
		The absolute tolerance for a zero singular value.  Singular values
		smaller than `atol` are considered to be zero.
	rtol : float
		The relative tolerance.	 Singular values less than rtol*smax are
		considered to be zero, where smax is the largest singular value.

	If both `atol` and `rtol` are positive, the combined tolerance is the
	maximum of the two; that is::
		tol = max(atol, rtol * smax)
	Singular values smaller than `tol` are considered to be zero.

	Return value
	------------
	ns : ndarray
		If `A` is an array with shape (m, k), then `ns` will be an array
		with shape (k, n), where n is the estimated dimension of the
		nullspace of `A`.  The columns of `ns` are a basis for the
		nullspace; each element in numpy.dot(A, ns) will be approximately
		zero.
	"""

	A = np.atleast_2d(A)
	u, s, vh = svd(A)
	tol = max(atol, rtol * s[0])
	nnz = (s >= tol).sum()
	ns = vh[nnz:].conj().T
	return ns

## https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
def qr_null(A, tol=None):
	from scipy.linalg import qr
	Q, R, P = qr(A.T, mode='full', pivoting=True)
	tol = np.finfo(R.dtype).eps if tol is None else tol
	rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
	return Q[:, rnk:].conj()
		
def intersect_flat( A, B ):
	if A.shape[0] == 0: return A
	if B.shape[0] == 0: return B
	null_A = qr_null( A ).T
	null_B = qr_null( B ).T
	
	assert( np.max( abs( np.dot(A,null_A.T) ) ) < 1e-10 )
	assert( np.max( abs( np.dot(B,null_B.T) ) ) < 1e-10 )
	
	null_A_union_B = np.vstack((null_A, null_B))
	return qr_null(null_A_union_B).T 
	
def intersect_flat( A, B ):
	if A.shape[0] == 0: return A
	if B.shape[0] == 0: return B
	null_A = qr_null( A ).T
	null_B = qr_null( B ).T
	
	assert( np.max( abs( np.dot(A,null_A.T) ) ) < 1e-10 )
	assert( np.max( abs( np.dot(B,null_B.T) ) ) < 1e-10 )
	
	null_A_union_B = np.vstack((null_A, null_B))
	return qr_null(null_A_union_B).T 

def convenient_intersect_flat( null_A, B ):
	'''
	null_A is a null space, B is a row space
	'''
	if B.shape[0] == 0: return B
	null_B = qr_null( B ).T
	
	assert( np.max( abs( np.dot(B,null_B.T) ) ) < 1e-10 )
	
	null_A_union_B = np.vstack((null_A, null_B))
	return qr_null(null_A_union_B).T
	
def intersect_all_flats( flats ):
	assert len( flats ) > 0
	null_union = qr_null( flats[0] ).T
	for A in flats[1:]:
		null_A = qr_null( A ).T
		null_union = np.vstack((null_union, null_A))
		
	return qr_null( null_union ).T

def pack( point, B ):
	'''
	`point` is a 12P-by-1 column matrix.
	`B` is a 12P-by-#handles matrix.
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
	X is a flattened array with (#handle+1)*12P entries.
	The first 12*P entries are `point`.
	The remaining entries are the 12P-by-#handles matrix B.
	
	where P = poses.
	'''
	point = X[:12*P].reshape(12*P, 1)
	B = X[12*P:].reshape(-1, 12*P).T

	return point, B

iteration = [0]
def show_progress( x ):
	iteration[0] += 1
	print("Iteration", iteration[0])
		
constraints = []
			
def optimize_flat_intersection(P, all_flats, seed):	
	## Goal function
	for H in range(1, MAX_H):
	
		x0 = np.ones((H+1)*12*P)
		x0[:12*P] = seed[0]
		 
		def f_flat_distance_sum(x):
			dist = 0
			pt, B = unpack(x,P)
			for j, flat in enumerate(all_flats):			
				intersect = convenient_intersect_flat(B.T, flat)
				dist += np.linalg.norm(np.dot(intersect, pt-seed[j]))
			
			return dist
			
		solution = scipy.optimize.minimize( f_flat_distance_sum, x0, constraints = constraints, callback = show_progress, options={'maxiter':10, 'disp':True} )
		if( abs( f_flat_distance_sum( solution.x ) ) < 1e-3 ):
			print("find #handles: ", H)
			return solution.x
			
	print("Exceed Maximum #handles ", MAX_H)
	return solution.x
	
def optimize_nullspace_directly(P, seed, row_mats, deformed_vs):

	def f_point_distance_sum(x):
		dist = 0
		pt, B = unpack(x,P)
		for j, vs in enumerate(deformed_vs):
			vprime = vs.reshape((3*P,1))	
			vbar = row_mats[j]
			lh = np.dot( np.dot(B.T, np.dot(vbar.T, vbar)), B )
			rh = np.dot( np.dot(B.T, vbar.T), np.dot(vbar, pt) - vprime )
			z = np.linalg.solve(lh, rh)
			z = z.reshape(-1,1)
			
			dist += np.linalg.norm(np.dot(vbar, pt+np.dot(B, z)) - vprime)
		
		return dist
	
	from flat_intersection_direct_gradients import f_and_dfdp_and_dfdB
	def f_point_distance_sum_and_gradient(x):
		pt, B = unpack(x,P)
		pt = pt.squeeze()
		
		f = 0
		grad_p = np.zeros(pt.shape)
		grad_B = np.zeros(B.shape)
		
		for j, vs in enumerate(deformed_vs):
			vprime = vs.ravel()
			vbar = row_mats[j]
			fj, gradj_p, gradj_B = f_and_dfdp_and_dfdB( pt, B, vbar, vprime )
			f += fj
			grad_p += gradj_p
			grad_B += gradj_B
		
		print( "f:", f )
		
		return f, pack( grad_p, grad_B )
	
	for H in range(4, MAX_H):
		print("#handles: ", H)
		x0 = np.random.rand((H+1)*12*P)
		x0[:12*P] = seed[0]
		
		## Without gradients:
		# solution = scipy.optimize.minimize( f_point_distance_sum, x0, constraints = constraints, callback = show_progress, options={'maxiter':10, 'disp':True} )
		## With gradients:
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, method = 'L-BFGS-B', callback = show_progress, options={'disp':True} )
		solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, callback = show_progress, options={'disp':True} )
		if( abs( f_point_distance_sum( solution.x ) ) < 1e-3 ):
			print("find #handles: ", H)
			return solution.x
			
	print("Exceed Maximum #handles ", MAX_H)
	return solution.x	
	
	
		
if __name__ == '__main__':
	if len(sys.argv) != 3:
		print( 'Usage:', sys.argv[0], 'path/to/rest_pose.obj', 'path/to/input.txt', file = sys.stderr )
		exit(-1)
	
	rest_mesh = TriMesh.FromOBJ_FileName( sys.argv[1] )
	deformed_vs = format_loader.load_poses( sys.argv[2] )
	
	## build flats
	assert( len(deformed_vs.shape) == 3 )
	P, N = deformed_vs.shape[0], deformed_vs.shape[1]
	deformed_vs = np.swapaxes(deformed_vs, 0, 1).reshape(N, P, 3)
	all_rights = deformed_vs.copy()
	
	all_flats = []
	all_R_mats = []
	for i, pos in enumerate(rest_mesh.vs):
		left_m = np.zeros((3*P, 12*P))
		unit_left_m = left_m.copy()
		pos_h = np.ones(4)
		pos_h[:3] = pos
		nm = np.linalg.norm( pos_h )
		for j in range(P):
			for k in range(3):
				unit_left_m[j*3+k, j*12+k*4:j*12+k*4+4] = pos_h/nm
				left_m[j*3+k, j*12+k*4:j*12+k*4+4] = pos_h
				
		all_rights[i] /= nm
		assert( np.allclose( np.dot(unit_left_m, unit_left_m.T), np.eye(3*P) ) )
		all_flats.append(unit_left_m)
		all_R_mats.append(left_m)
	all_flats = np.array( all_flats )
	all_R_mats = np.array( all_R_mats )
	
	print( "The rank of the stack of all pose row matrices is: ", np.linalg.matrix_rank( np.vstack( all_flats ) ) )
	
	## find flat intersection
# 	start_time = time.time()
# 	intersect = all_flats[0]
# 	for flat in all_flats[1:]:
# 		intersect = intersect_flat( intersect, flat )
# 	print( "Time for intersect all flats iteratively: ", time.time() - start_time )

	seed = []
	for i in range( N ):
		for j in range( P ):
			dist = (deformed_vs[i,j] - rest_mesh.vs[i])[:,np.newaxis]
			translation = np.hstack((np.eye(3), dist)).reshape(-1)
			seed.append( translation )
	seed = np.array( seed ).reshape( N, -1 )	
	assert( np.max( abs( np.dot(all_flats[0], seed[0].reshape(-1,1)) - all_rights[0].reshape(-1,1) ) ) < 1e-10 )
	
# 	optimize_flat_intersection(P, all_flats, seed)
	optimize_nullspace_directly(P, seed, all_R_mats, deformed_vs)


		
