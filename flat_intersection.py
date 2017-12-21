"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division

import os
import sys
import argparse
import time
import glob
# import numpy as np
import autograd.numpy as np
from autograd import grad
from numpy.linalg import svd
import scipy.optimize

np.set_printoptions( linewidth = 2000 )

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
	The first 12*P entries are `point`.
	The remaining entries are the 12P-by-#(handles-1) matrix B.
	
	where P = poses.
	'''
	point = X[:12*P].reshape(12*P, 1)
	B = X[12*P:].reshape(-1,12*P).T

	return point, B

iteration = [0]
def reset_progress():
	iteration[0] = 0
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
			
		reset_progress()
		solution = scipy.optimize.minimize( f_flat_distance_sum, x0, constraints = constraints, callback = show_progress, options={'maxiter':10, 'disp':True} )
		if( abs( f_flat_distance_sum( solution.x ) ) < 1e-3 ):
			print("find #handles: ", H)
			return solution.x
			
	print("Exceed Maximum #handles ", MAX_H)
	return solution.x

def zero_energy_test(base_dir):
	def load_perbone_tranformation( path ):	
		with open( path ) as f:
			v = []
			for i, line in enumerate( f ):
				v = v + list( map( float, line.strip().split() ) )
		 
			M = np.array(v)
	
		M = M.reshape( -1, 4, 3 )
		MT = np.swapaxes(M, 1, 2)
		MT = MT.reshape(-1, 12)
	
		return MT

	gt_bone_paths = glob.glob(base_dir + "/*.Tmat")
	gt_bone_paths.sort()
	gt_bones = np.array([ load_perbone_tranformation(transform_path) for transform_path in gt_bone_paths ])
	gt_bones = np.swapaxes(gt_bones, 0, 1)
	gt_bones = gt_bones.reshape( len(gt_bones), -1 )
	
	
	def load_pervertex_transformation( path ):
		with open( path ) as f:
			for i, line in enumerate( f ):
				if i == 0:
					dims = list( map( int, line.strip().split() ) )
					M = np.zeros( np.prod( dims ) )
			
				else:
					M[i-1] = float( line )
	
		M = M.reshape( dims ).T.reshape(-1,4,3)
		MT = np.swapaxes(M, 1, 2)
		MT = MT.reshape(-1, 12)
	
		return MT
	
	gt_vertex_paths = glob.glob(base_dir + "/*.DMAT")
	gt_vertex_paths.sort()
	gt_vertices = np.array([ load_pervertex_transformation(path) for path in gt_vertex_paths ])
	gt_vertices = np.swapaxes(gt_vertices, 0, 1)
	gt_vertices = gt_vertices.reshape( len(gt_vertices), -1 )
	
	gt_W_path = "models/cheburashka/cheburashka.DMAT"
	gt_W = np.array( format_loader.load_DMAT(gt_W_path) ).T
	
	return gt_bones, gt_vertices, gt_W
	
# def zero_energy_test(base_dir):
# 	gt_bone_paths = glob.glob(base_dir + "*.DMAT")
# 	gt_bone_paths.sort()
# 	gt_bones = np.array([ format_loader.load_Tmat(transform_path) for transform_path in gt_bone_paths ])
# 	gt_bones = np.swapaxes(gt_bones, 0, 1)
# 	gt_bones = gt_bones.reshape( len(gt_bones), -1 )
# 	
# 	return gt_bones
	
def optimize_nullspace_directly(P, H, row_mats, deformed_vs, ground_truth_path = None, recovery_test = None ):

	## 0 energy test
	if ground_truth_path is not None:
		gt_bones, gt_vertices, gt_W = zero_energy_test(ground_truth_path)

	## To make function values comparable, we need to normalize.
	xyzs = np.asarray([ row_mat[0,:3] for row_mat in row_mats ])
	diag = xyzs.max( axis = 0 ) - xyzs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	normalization = 1./( len(row_mats) * diag )
	
	def f_point_distance_sum(x):
		dist = 0
		pt, B = unpack(x,P)
		num_underconstrained = 0
		for j, vs in enumerate(deformed_vs):
			vprime = vs.reshape((3*P,1))	
			vbar = row_mats[j]
			vB = np.dot( vbar, B )
			lh = np.dot( vB.T, vB )
			rh = -np.dot( vB.T, np.dot(vbar, pt) - vprime )
			## lh should be well-behaved. Its smallest singular value should not be zero.
			if np.linalg.svd( lh, compute_uv = False )[-1] < 1e-7:
				num_underconstrained += 1
				# z = np.dot( np.linalg.pinv(lh), rh )
				## Solve still works better without complaining. I guess it's not zero enough.
				z = np.linalg.solve(lh, rh)
			else:
				z = np.linalg.solve(lh, rh)
			z = z.reshape(-1,1)
			
			# assert( abs(np.dot( x.reshape(-1,12).T, gt_W[j] ).ravel() - gt_vertices[j]).max() < 1e-6 )
			# assert( abs((pt+np.dot(B, z)).ravel() - gt_vertices[j]).max() < 1e-4 )
			dist += np.linalg.norm(np.dot(vbar, pt+np.dot(B, z)) - vprime)
			# dist += np.linalg.norm(np.dot(vbar, gt_vertices[j]) - vprime)
	
		# print( "f:", dist )
		# if num_underconstrained > 0:
		print( "Underconstrained vertices:", num_underconstrained )
		
		return dist * normalization
		
	def f_point_distance_sum_gradient(x):
		g = grad(f_point_distance_sum,0)
		return g(x)
		
	def f_sum_and_gradient(x):
		return 	f_point_distance_sum(x), f_point_distance_sum_gradient(x)
		
	from flat_intersection_direct_gradients import f_and_dfdp_and_dfdB, f_and_dfdp_and_dfdB_dumb
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
			# fj2, gradj_p2, gradj_B2 = f_and_dfdp_and_dfdB_dumb( pt, B, vbar, vprime )
			# assert "f close?:" and abs( fj - fj2 ).max() < 1e-6
			# assert "gp close?:" and abs( gradj_p - gradj_p2 ).max() < 1e-6
			# assert "gB close?:" and abs( gradj_B - gradj_B2 ).max() < 1e-6
			f += fj
			grad_p += gradj_p
			grad_B += gradj_B
		
		# print( "f:", f )
		
		return f * normalization, pack( grad_p * normalization, grad_B * normalization )
	
	print("#handles: ", H)
	x0 = np.random.rand(H*12*P)
	pt = np.zeros((3,4))
	pt[:,:3] = np.eye(3)
	pt = pt.ravel()
	for i in range(P):
		x0[12*i: 12*(i+1)] = pt
	pt, B = unpack( x0, P )
	for i in range(B.shape[1]):
		B[:,i] /= np.linalg.norm(B[:,i])
	x0 = pack( pt, B )
	print( "Initial pt.T: ", pt.T )
	print( "Initial B.T: ", repr(B.T) )
	
	## zero energy test for cube4
	if ground_truth_path is not None:
		pt = gt_bones[0]
		B = (gt_bones[1:] - pt).T
		x0 = pack( pt, B )
		## Recovery test
		if recovery_test is not None:
			x0 += + recovery_test*x0
		
		print( "There are", len(gt_bones), "ground truth bones." )
		print( "If they are linearly independent, then the following has no zeros:", np.linalg.svd( B.T, compute_uv = False ) )
	
	## Without gradients:
	print( "f_point_distance_sum value at x0:", f_point_distance_sum( x0 ) )
	reset_progress()
	# solution = scipy.optimize.minimize( f_sum_and_gradient, x0, jac = True, constraints = constraints, callback = show_progress, options={'maxiter':10, 'disp':True} )
	# solution = scipy.optimize.minimize( f_sum_and_gradient, x0, jac = True, constraints = constraints, method = 'L-BFGS-B', callback = show_progress, options={'maxiter':10, 'disp':True} )
	## With gradients:
	print( "f_point_distance_sum_and_gradient value at x0:", f_point_distance_sum_and_gradient( x0 )[0] )
	reset_progress()
	# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, method = 'L-BFGS-B', callback = show_progress, options={'disp':True} )
	solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, callback = show_progress, options={'disp':True} )
	# grad_err = scipy.optimize.check_grad( lambda x: f_point_distance_sum_and_gradient(x)[0], lambda x: f_point_distance_sum_and_gradient(x)[1], x0 )
	# print( "scipy.optimize.check_grad() error:", grad_err )
	
	converged = abs( f_point_distance_sum( solution.x ) ) < 1e-2
			
	return converged, solution.x	
		
if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser( description='Solve for transformation subspace.' )
	parser.add_argument( 'rest_pose', type=str, help='Rest pose (OBJ).')
	parser.add_argument( 'deformed_vs', type=str, help='Deformed vertices.')
	parser.add_argument('--handles', '--H', type=int, help='Number of handles.')
	parser.add_argument('--ground-truth', '--GT', type=str, help='Ground truth data path.')
	parser.add_argument('--recovery', '--R', type=float, help='Recovery test epsilon (default no recovery test).')
	
	args = parser.parse_args()
	H = args.handles
	ground_truth_path = args.ground_truth
	recovery_test = args.recovery
	
	rest_mesh = TriMesh.FromOBJ_FileName( args.rest_pose )
	deformed_vs = format_loader.load_poses( args.deformed_vs )
	
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
				left_m     [j*3+k, j*12+k*4:j*12+k*4+4] = pos_h
				
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
	if H is None:
		Hs = range(2, MAX_H)
	else:
		Hs = [H]
	for H in Hs:
		converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, ground_truth_path = ground_truth_path, recovery_test = recovery_test )
		if ground_truth_path is not None and converged:
			print("Converged at handle #", H)
			exit(0)
	
	if ground_truth_path is not None:
		print("Exceed Maximum #handles ", MAX_H)

