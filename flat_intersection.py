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
import numpy as np
# import autograd.numpy as np
# from autograd import grad
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
	The first 12*P entries are `point` as a 12*P-by-1 matrix.
	The remaining entries are the 12P-by-#(handles-1) matrix B.
	
	where P = poses.
	'''
	P = poses
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
	
# 	gt_W_path = "models/cheburashka/cheburashka.DMAT"
# 	gt_W = np.array( format_loader.load_DMAT(gt_W_path) ).T
	
	return gt_bones, gt_vertices
	
# def zero_energy_test(base_dir):
#	gt_bone_paths = glob.glob(base_dir + "*.DMAT")
#	gt_bone_paths.sort()
#	gt_bones = np.array([ format_loader.load_Tmat(transform_path) for transform_path in gt_bone_paths ])
#	gt_bones = np.swapaxes(gt_bones, 0, 1)
#	gt_bones = gt_bones.reshape( len(gt_bones), -1 )
#	
#	return gt_bones

def normalization_factor_from_row_mats( row_mats ):
	## To make function values comparable, we need to normalize.
	xyzs = np.asarray([ row_mat[0,:3] for row_mat in row_mats ])
	return normalization_factor_from_xyzs( xyzs )
def normalization_factor_from_xyzs( xyzs ):
	## To make function values comparable, we need to normalize.
	diag = xyzs.max( axis = 0 ) - xyzs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	normalization = 1./( len(xyzs) * diag )
	print( "Normalization of 1/(bounding box diagonal * num-vertices):", normalization )
	return normalization

def optimize_nullspace_directly(P, H, row_mats, deformed_vs, x0, strategy = None, max_iter = None):
	
	## To make function values comparable, we need to normalize.
	normalization = normalization_factor_from_row_mats( row_mats )
	
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
			d = (np.dot(vbar, pt+np.dot(B, z)) - vprime).ravel()
			dist += np.dot(d,d)
			# dist += np.linalg.norm(np.dot(vbar, pt+np.dot(B, z)) - vprime)
			# dist += np.linalg.norm(np.dot(vbar, gt_vertices[j]) - vprime)
	
		# print( "f:", dist )
		# if num_underconstrained > 0:
		print( "Underconstrained vertices:", num_underconstrained )
		
		return dist * normalization
		
	def f_point_distance_sum_gradient(x):
		g = grad(f_point_distance_sum,0)
		return g(x)
		
	def f_sum_and_gradient(x):
		return	f_point_distance_sum(x), f_point_distance_sum_gradient(x)
		
	from flat_intersection_direct_gradients import f_and_dfdp_and_dfdB, f_and_dfdp_and_dfdB_dumb, fAndGpAndHp_fast, d2f_dp2_dumb
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
			
			## test hessian_p and fast version.
			# # fj_fast, gradj_p_fast, hessj_p_fast = fAndGpAndHp_fast( pt, B, vbar, vprime )
			# hessj_p = d2f_dp2_dumb( pt, B, vbar, vprime )
			# assert "f close?:" and abs( fj - fj_fast ).max() < 1e-6
			# assert "gp close?:" and abs( gradj_p - gradj_p_fast ).max() < 1e-6
			# assert "gB close?:" and abs( hessj_p - hessj_p_fast ).max() < 1e-6
			
			# fj2, gradj_p2, gradj_B2 = f_and_dfdp_and_dfdB_dumb( pt, B, vbar, vprime )
			# assert "f close?:" and abs( fj - fj2 ).max() < 1e-6
			# assert "gp close?:" and abs( gradj_p - gradj_p2 ).max() < 1e-6
			# assert "gB close?:" and abs( gradj_B - gradj_B2 ).max() < 1e-6
			f += fj
			grad_p += gradj_p
			grad_B += gradj_B
		
		print( "f:", f )
		
		return f * normalization, pack( grad_p * normalization, grad_B * normalization )
	
	def f_hess(x):
		import flat_intersection_hessians
		
		print( "Computing hessian begins." )
		
		hess = np.zeros( ( len(x), len(x) ) )
		for j, vs in enumerate(deformed_vs):
			vprime = vs.ravel()
			vbar = row_mats[j]
			hess += flat_intersection_hessians.hess( x, vbar, vprime, P )
		
		print( "Computing hessian finished." )
		
		return hess
	
	def f_hess_onlyp(x):
		pt, B = unpack(x,P)
		pt = pt.squeeze()
		
		hess = np.eye( len(x) )
		hess[:len(pt), :len(pt)] = 0.
		for j, vs in enumerate(deformed_vs):
			vprime = vs.ravel()
			vbar = row_mats[j]
			_, _, hessj_p = fAndGpAndHp_fast( pt, B, vbar, vprime )
			hess[:len(pt),:len(pt)] += hessj_p
		
		return hess
	
	## Print the initial function.
	print( "f_point_distance_sum value at x0:", f_point_distance_sum( x0 ) )
	print( "f_point_distance_sum_and_gradient value at x0:", f_point_distance_sum_and_gradient( x0 )[0] )
	
	## Optimize the quadratic degrees-of-freedom once.
	## (It helps a lot once. It helps a lot less in future iterations.)
	## UPDATE: Actually, this is harmful! This followed by the gradient approach slows things down!
	# _, x0 = optimize_approximated_quadratic( P, H, row_mats, deformed_vs, x0, max_iter = 1 )
	
	reset_progress()
	## strategies: 'function', 'gradient', 'hessian', 'mixed'
	if strategy is None: strategy = 'gradient'
	if strategy == 'function':
		## Without gradients (or autograd):
		# solution = scipy.optimize.minimize( f_sum_and_gradient, x0, jac = True, constraints = constraints, callback = show_progress, options={'maxiter':10, 'disp':True} )
		# solution = scipy.optimize.minimize( f_sum_and_gradient, x0, jac = True, constraints = constraints, method = 'L-BFGS-B', callback = show_progress, options={'maxiter':10, 'disp':True} )
		solution = scipy.optimize.minimize( f_point_distance_sum, x0, constraints = constraints, method = 'L-BFGS-B', callback = show_progress, options={'maxiter':max_iter, 'disp':True} )
	elif strategy == 'gradient':
		## With gradients:
		## check_grad() is too slow to always run.
		# grad_err = scipy.optimize.check_grad( lambda x: f_point_distance_sum_and_gradient(x)[0], lambda x: f_point_distance_sum_and_gradient(x)[1], x0 )
		# print( "scipy.optimize.check_grad() error:", grad_err )
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, method = 'L-BFGS-B', callback = show_progress, options={'disp':True} )
		solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, callback = show_progress, options={'maxiter':max_iter, 'disp':True} )
	elif strategy == 'hessian':
		## Use the Hessian:
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, hess = f_hess, method = 'Newton-CG', callback = show_progress, options={'disp':True} )
		solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, hess = f_hess, method = 'Newton-CG', callback = show_progress, options={'maxiter':max_iter, 'disp':True} )
	elif strategy == 'mixed':
		## Mixed with quadratic for p:
		x = x0.copy()
		MAX_NONLINEAR_ITER = 10
		while True:
			## Optimize quadratic once (more than that is wasted).
			_, x = optimize_approximated_quadratic( P, H, row_mats, deformed_vs, x, max_iter = 1 )
			solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x, jac = True, method = 'L-BFGS-B', callback = show_progress, options={'disp':True, 'maxiter': MAX_NONLINEAR_ITER} )
			x = solution.x
			if solution.success: break
	elif strategy == 'grassmann':
		## Mixed with grassmann projections
		import flat_intersection_cayley_grassmann_gradients as grassmann
		x = x0.copy()
		MAX_NONLINEAR_ITER = 100
		MIN_NONLINEAR_ITER = 10
		while True:
			
			p, B = unpack( x, P )
			A = grassmann.A_from_non_Cayley_B( B )
			B = grassmann.B_from_Cayley_A( A, H )
			x = pack( p, B )
			
			solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x, jac = True, method = 'BFGS', callback = show_progress, options={'disp':True, 'maxiter': MAX_NONLINEAR_ITER} )
			x = solution.x
			## Only break if we converge and ran just a few iterations.
			if solution.success and solution.nit < MIN_NONLINEAR_ITER: break
	else:
		raise RuntimeError( "Unknown strategy: " + str(strategy) )
	
	f = f_point_distance_sum_and_gradient( solution.x )[0]
	print( "f_point_distance_sum_and_gradient value at solution:", f )
	converged = abs( f ) < 1e-2
	
	return converged, solution.x

def optimize_nullspace_cayley(P, H, row_mats, deformed_vs, x0, strategy = None):

	## To make function values comparable, we need to normalize.
	normalization = normalization_factor_from_row_mats( row_mats )
	
	if True:
		## This one is better. Fewer degrees of freedom.
		import flat_intersection_cayley_grassmann_gradients as cayley
	else:
		import flat_intersection_cayley_gradients as cayley
	
	p, B = unpack( x0, P )
	A = cayley.A_from_non_Cayley_B( B )
	x0 = cayley.pack( p, A, P, H )
	
	## start from zero
	## UPDATE: This doesn't work. There is a singularity in the gradient.
	# x0 *= 0.
	# x0 += 1e-5
	## start from random
	# x0 = np.random.random(len(x0))
	
	def f_point_distance_sum_and_gradient(x):
		pt, A = cayley.unpack( x, P, H )
		pt = pt.squeeze()
		
		f = 0
		grad_p = np.zeros(pt.shape)
		grad_A = np.zeros(A.shape)
		
		for j, vs in enumerate(deformed_vs):
			vprime = vs.ravel()
			vbar = row_mats[j]
			fj, gradj_p, gradj_A = cayley.f_and_dfdp_and_dfdA( pt, A, vbar, vprime, H )
			
			f += fj
			grad_p += gradj_p
			grad_A += gradj_A
		
		print( "f:", f )
		
		return f * normalization, cayley.pack( grad_p * normalization, grad_A * normalization, P, H )
	
	## Print the initial function.
	print( "f_point_distance_sum_and_gradient value at x0:", f_point_distance_sum_and_gradient( x0 )[0] )
	
	reset_progress()
	## strategies: 'function', 'gradient', 'hessian', 'mixed'
	if strategy is None: strategy = 'gradient'
	if strategy == 'function':
		## Without gradients:
		solution = scipy.optimize.minimize( lambda x: f_point_distance_sum_and_gradient( x )[0], x0, constraints = constraints, method = 'L-BFGS-B', callback = show_progress, options={'maxiter':10, 'disp':True} )
	elif strategy == 'gradient':
		## With gradients:
		## check_grad() is too slow to always run.
		# grad_err = scipy.optimize.check_grad( lambda x: f_point_distance_sum_and_gradient(x)[0], lambda x: f_point_distance_sum_and_gradient(x)[1], x0 )
		# print( "scipy.optimize.check_grad() error:", grad_err )
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, method = 'L-BFGS-B', callback = show_progress, options={'disp':True} )
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, method = 'CG', callback = show_progress, options={'disp':True} )
		solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, method = 'BFGS', callback = show_progress, options={'disp':True} )
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, callback = show_progress, options={'disp':True} )
	else:
		raise RuntimeError( "Unknown strategy: " + str(strategy) )
	
	f = f_point_distance_sum_and_gradient( solution.x )[0]
	print( "f_point_distance_sum_and_gradient value at solution:", f )
	converged = abs( f ) < 1e-2
	
	pt, A = cayley.unpack( solution.x, P, H )
	B = cayley.B_from_Cayley_A( A, H )
	x_pb = pack( pt, B )
	
	return converged, x_pb

def optimize_approximated_quadratic(P, H, row_mats, deformed_vs, x0, f_eps = None, x_eps = None, max_iter = None):

	## To make function values comparable, we need to normalize.
	xyzs = np.asarray([ row_mat[0,:3] for row_mat in row_mats ])
	diag = xyzs.max( axis = 0 ) - xyzs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	normalization = 1./( len(row_mats) * diag )
	
	def unpack_shifted( x, P, j ):
		pt, B = unpack( x, P )
		## Make sure pt is a column matrix.
		pt = pt.squeeze()[...,np.newaxis]
		assert len( pt.shape ) == 2
		assert pt.shape[1] == 1
		
		## Switch pt with the j-th column of [p;B].
		## 1 Convert [pt;B] from an origin and basis to a set of points.
		W = np.hstack((pt,pt+B))
		## 2 Get the j-th column as pt.
		pt = W[:,j:j+1].copy()
		## 3 Take the remaining columns - pt.
		W -= pt
		B = np.hstack( ( W[:,:j], W[:,j+1:] ) )
		
		return pt, B
	
	def pack_shifted( pt, B, j ):
		## Make sure pt is a column matrix.
		pt = pt.squeeze()[...,np.newaxis]
		assert len( pt.shape ) == 2
		assert pt.shape[1] == 1
		
		## Convert pt + B[:,:j], pt, pt + B[:,j+1:] to a set of points.
		W = np.hstack(( pt + B[:,:j], pt, pt + B[:,j:] ))
		pt = W[:,0:1]
		B = W[:,1:] - pt
		return pack( pt, B )
	
	## Verify that we can unpack and re-pack shifted without changing anything.
	assert abs( pack_shifted( *( list( unpack_shifted( np.arange(36), 1, 1 ) ) + [ 1 ] ) ) - np.arange(36) ).max() < 1e-10
	# assert abs( pack_shifted( *( list( unpack_shifted( x0, P, 1 ) ) + [ 1 ] ) ) - x0 ).max() < 1e-10
	
	def f_point_distance_sum(x):
		f = 0
		x = x.copy()
		
		# for j in range(H):
		## Only once. More than that doesn't help.
		for j in range(1):
			pt, B = unpack_shifted( x, P, j )
			
			Q = np.zeros((12*P, 12*P))
			L = np.zeros(12*P)
			C = 0
			for i, vs in enumerate(deformed_vs):
				vprime = vs.reshape((3*P,1))
				vbar = row_mats[i]
				vB = np.dot( vbar, B )
				vBBv = np.dot( vB.T, vB )
				inv_vBBv = np.linalg.pinv(vBBv)
				A_i = np.dot(np.dot(vB, inv_vBBv), vB.T)
			
				foo = np.eye(3*P)-A_i
				S_i = np.dot(foo, vbar)
				r_i = np.dot(foo, vprime).ravel()

				Q += np.dot(S_i.T, S_i)
				L += np.dot(S_i.T, r_i)
				C += np.dot(r_i.T, r_i)
			
			f = normalization * (np.dot(np.dot(pt.T,Q), pt) - 2*np.dot(pt.T,L) + C)
			pt_new = np.linalg.solve(Q,L)
			f_new = normalization * (np.dot(np.dot(pt_new.T,Q), pt_new) - 2*np.dot(pt_new.T,L) + C)
			
			x = pack_shifted( pt_new, B, j )
			
			print( "Sub-iteration", j, "finished. Old function value:", f )
			print( "Sub-iteration", j, "finished. New function value:", f_new )
#			print( "New x value:", xmat.reshape(-1) )
		#res_x = x.copy()
		#res_x[:12*P] = xmat[-1]
		#res_x[12*P:] = xmat[:-1].reshape(-1)
		
		print( "Finished iteration." )
		return f, x
	
	if f_eps is None:
		f_eps = 1e-10
	if x_eps is None:
		x_eps = 1e-10
	if max_iter is None:
		max_iter = 9999
	
	f_prev = None
	x_prev = x0.copy()
	iterations = 0
	converged = False
	while( True ):
		iterations += 1
		if iterations > max_iter:
			print( "Terminating due to too many iterations:", max_iter )
			break
		
		print( "Starting iteration", iterations )
		f, x = f_point_distance_sum(x_prev)
		## If this is the first iteration, pretend that the old function value was
		## out of termination range.
		if f_prev is None: f_prev = f + 100*f_eps
		
		if f - f_prev > 0:
			print( "WARNING: Function value increased." )
		if abs( f_prev - f ) < f_eps:
			print( "Function change too small, terminating:", f_prev - f )
			converged = True
			break
		x_change = abs( x_prev - x ).max()
		if x_change < x_eps:
			print( "Variables change too small, terminating:", x_change )
			converged = True
			break
		
		f_prev = f
		x_prev = x.copy()
	
	print( "Terminated after", iterations, "iterations." )
	
	return converged, x

def optimize_biquadratic( P, H, row_mats, deformed_vs, x0, solve_for_rest_pose = False, f_eps = None, x_eps = None, max_iter = None, f_zero_threshold = None, strategy = None, W_projection = None, z_strategy = None, **kwargs ):
	'''
	If solve_for_rest_pose is False (the default), returns ( converged, final x ).
	If solve_for_rest_pose is True, returns ( converged, final x, and updated row_mats ).
	'''
	
	## To make function values comparable, we need to normalize.
	xyzs = np.asarray([ row_mat[0,:3] for row_mat in row_mats ])
	diag = xyzs.max( axis = 0 ) - xyzs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	# normalization = 1./( len(row_mats) * diag )
	normalization = normalization_factor_from_row_mats( row_mats )
	## We don't want this uniform per-vertex normalization because
	## we may do special weight handling.
	normalization *= len( row_mats )
	print( "Normalization:", normalization )
	
	if strategy is None:
		strategy = ['ssv:weighted']
	else:
		strategy = strategy.split('+')
	
	print( "Strategy:", strategy )
	
	use_pseudoinverse = False
	if 'pinv' in strategy:
		use_pseudoinverse = True
	
	def unpack_W( x, P ):
		pt, B = unpack( x, P )
		## Make sure pt is a column matrix.
		pt = pt.squeeze()[...,np.newaxis]
		assert len( pt.shape ) == 2
		assert pt.shape[1] == 1
		
		## Switch pt with the j-th column of [p;B].
		## 1 Convert [pt;B] from an origin and basis to a set of points.
		W = np.hstack((pt,pt+B))
		
		return W
	
	def pack_W( W ):
		pt = W[:,0:1]
		B = W[:,1:] - pt
		return pack( pt, B )
	
	## Verify that we can unpack and re-pack shifted without changing anything.
	assert abs( pack_W( unpack_W( np.arange(36), 1 ) ) - np.arange(36) ).max() < 1e-10
	
	if f_eps is None:
		f_eps = 1e-6
	if x_eps is None:
		## To make xtol approximately match scipy's default gradient tolerance (gtol) for BFGS.
		x_eps = 1e-4
	if max_iter is None:
		max_iter = 9999
	if f_zero_threshold is None:
		f_zero_threshold = 0.0
	
	print( "optimize_biquadratic() with strategy:", strategy, "f_eps:", f_eps, "x_eps:", x_eps, "max_iter:", max_iter, "f_zero_threshold:", f_zero_threshold, "W_projection:", W_projection, "z_strategy:", z_strategy )
	
	import flat_intersection_biquadratic_gradients as biquadratic
	
	first_column = None
	if W_projection == 'first':
		def estimate_point_on_subspace( guess_data, guess_errors, guess_ssv ):
			## Use the inverse of the error as the weight.
			weights = 1./(1e-5 + guess_errors)
			## Ignore points with bad smallest singular values.
			weights[ guess_ssv < 1e-8 ] = 0.
			## If those points are reliable, then the weighted average of them must also
			## lie on the simplex.
			first_column = np.average( guess_data, axis = 0, weights = weights )
			return first_column
		def replace_x0_with_better_p( x0, poses, first_column ):
			W = unpack_W( x0.copy(), poses )
			W[:,0] = first_column
			x0 = pack_W( W )
			return x0
		
		first_column = estimate_point_on_subspace( kwargs['guess_data'], kwargs['guess_errors'], kwargs['guess_ssv'] )
		x0 = replace_x0_with_better_p( x0, P, first_column )
	
	f_prev = None
	W_prev = unpack_W( x0.copy(), P )
	iterations = 0
	converged = False
	try:
		while( True ):
			iterations += 1
			if iterations > max_iter:
				print( "Terminating due to too many iterations: ", max_iter )
				break
			
			print( "Starting iteration", iterations )
			
			## 1 Find the optimal z.
			## (optional) 2 Solve for the new V.
			## 3 Accumulate the linear matrix equation for W.
			## 4 Solve for the new W.
			
			f = 0
			weights = 0
			
			fis = np.zeros( len( deformed_vs ) )
			As = []
			Bs = []
			Ys = []
			
			## If we are solving for the rest pose, make a copy of row_mats, because
			## we will modify it.
			if solve_for_rest_pose:
				row_mats = row_mats.copy()
			
			for i, vs in enumerate(deformed_vs):
				vprime = vs.reshape((3*P,1))
				vbar = row_mats[i]
				
				## 1
				z, ssz, fi = biquadratic.solve_for_z( W_prev, vbar, vprime, return_energy = True, use_pseudoinverse = use_pseudoinverse, strategy = z_strategy )
				fis[i] = fi
				
				if 'ssv:skip' in strategy:
					if ssz < 1e-5:
						print( "Skipping vertex with small singular value this time" )
						continue
					else:
						ssz = 1.0
				elif 'ssv:weighted' in strategy:
					pass
				else:
					ssz = 1.0
				
				## 2
				if solve_for_rest_pose:
					Q,L,C = biquadratic.quadratic_for_V( W_prev, z, vprime )
					v = vbar[:3,0]
					old_fi = np.dot( np.dot( v, Q ), v ) + np.dot( L, v ) + C
					
					vbar, fi = biquadratic.solve_for_V( W_prev, z, vprime, return_energy = True, use_pseudoinverse = use_pseudoinverse )
					## Store the new vbar.
					row_mats[i] = vbar
					fis[i] = fi
				
				## 3
				A, B, Y = biquadratic.linear_matrix_equation_for_W( vbar, vprime, z )
				As.append( ssz*A )
				Bs.append( B )
				Ys.append( ssz*Y )
				
				f += fi * ssz
				weights += ssz
			
			## 4
			W = biquadratic.solve_for_W( As, Bs, Ys, use_pseudoinverse = use_pseudoinverse, projection = W_projection, first_column = first_column )
			f *= normalization / weights
			print( "Function value:", f )
			print( "Max sub-function value:", fis.max() )
			print( "Min sub-function value:", fis.min() )
			print( "Average sub-function value:", np.average( fis ) )
			print( "Median sub-function value:", np.median( fis ) )
			
			## If this is the first iteration, pretend that the old function value was
			## out of termination range.
			if f_prev is None: f_prev = f + 100*f_eps
			
			if f - f_prev > 0:
				print( "WARNING: Function value increased." )
			if abs( f_prev - f ) < f_eps:
				print( "Function change too small, terminating:", f_prev - f )
				converged = True
				break
			## To make xtol approximately match scipy's default gradient tolerance (gtol) for BFGS,
			## use norm() instead of the max change.
			x_change_norm = np.linalg.norm( W_prev - W )
			x_change_max = abs( W_prev - W ).max()
			print( "x change (norm):", x_change_norm )
			print( "x change (max):", x_change_max )
			x_change = x_change_norm
			if x_change < x_eps:
				print( "Variables change too small, terminating:", x_change )
				converged = True
				break
			if f < f_zero_threshold:
				print( "Function below zero threshold, terminating." )
				converged = True
				break
			
			f_prev = f
			W_prev = W.copy()
	
	except KeyboardInterrupt:
		print( "Terminated by KeyboardInterrupt." )
	
	print( "Terminated after", iterations, "iterations." )
	
	if solve_for_rest_pose:
		return converged, pack_W( W ), row_mats
	else:
		return converged, pack_W( W )

def optimize_laplacian( P, H, rest_mesh, deformed_vs, qs_data, qs_errors, qs_ssv, f_eps = None, x_eps = None, max_iter = None, f_zero_threshold = None, z_strategy = None ):
	'''
	Returns ( converged, final x ).
	'''
	
	vs = np.asfarray( rest_mesh.vs )
	## To make function values comparable, we need to normalize.
	E_data_weight = 1.*normalization_factor_from_xyzs( vs )/P
	E_local_weight = 1./(len(vs)*P)
	E_input_weight = 1.0 ## See strategy = 'lsq' below.
	print( "E_data_weight:", E_data_weight )
	print( "E_local_weight:", E_local_weight )
	print( "E_input_weight:", E_input_weight )
	
	if f_eps is None:
		f_eps = 1e-6
	if x_eps is None:
		## To make xtol approximately match scipy's default gradient tolerance (gtol) for BFGS.
		x_eps = 1e-4
	if max_iter is None:
		max_iter = 9999
	if f_zero_threshold is None:
		f_zero_threshold = 0.0
	
	graph_laplacian = 'once'
	# graph_laplacian = 'always'
	# graph_laplacian = 'never'
	
	strategy = None
	# strategy = 'lsq'
	# strategy = 'lerp'
	
	print( "optimize_laplacian():", "strategy:", strategy, "f_eps:", f_eps, "x_eps:", x_eps, "max_iter:", max_iter, "f_zero_threshold:", f_zero_threshold, "z_strategy:", z_strategy )
	
	import flat_intersection_laplacian as laplacian
	
	deformed_vs = deformed_vs.reshape(-1,3*P)
	
	assert len( vs ) == len( deformed_vs )
	assert vs.shape[1] == 3
	assert deformed_vs.shape[1] == 3*P
	
	## This never changes. Precompute it.
	E_data = laplacian.quadratic_for_E_data( vs, deformed_vs )
	num_vertices = len( vs )
	neighbors = [ np.asarray( rest_mesh.vertex_vertex_neighbors(i) ) for i in range(num_vertices) ]
	poses = P
	
	## We should have an initial guess transformation for each point.
	assert len( qs_data ) == len( vs )
	assert qs_data.shape[1]%12 == 0
	## We should have error and smallest singular value information for the guesses.
	assert len( qs_data ) == len( qs_errors )
	assert len( qs_data ) == len( qs_ssv )
	## Errors and ssv should be 1D arrays.
	qs_errors = qs_errors.squeeze()
	qs_ssv = qs_ssv.squeeze()
	assert len( qs_errors.shape ) == 1
	assert len( qs_ssv.shape ) == 1
	## Errors should be positive
	assert ( qs_errors > -1e-10 ).all()
	## We assume that there is a large variation in error, so the median should be far from 0.
	assert abs( np.median(qs_errors) ) > 1e-5
	## Convert error into [0,1] lerp values (0 error is very confident so 0,
	## median error is not confidence so 1).
	lerpval = ( qs_errors/np.median(qs_errors) ).clip( 0.0, 1.0 )
	## Small singular values mean no confidence, too.
	lerpval[qs_ssv<1e-8] = 1.0
	## Make lerpval a column matrix for broadcasting.
	assert len( lerpval.shape ) == 1
	lerpval = lerpval.reshape(-1,1)
	
	plot = False
	if plot:
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import axes3d
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
		ax = fig.gca(projection='3d')
		from space_mapper import SpaceMapper
		mapper = SpaceMapper.Uncorrellated_Space( qs_data, dimension = 3 )
		
		Ts3D = mapper.project( qs_data )
		ax.scatter( Ts3D.T[0], Ts3D.T[1], Ts3D.T[2] )
		plt.show()
		plt.pause(2)
	
	f_prev = None
	Ts = qs_data.copy()
	Ts_prev = Ts.copy()
	iterations = 0
	converged = False
	try:
		while( True ):
			iterations += 1
			if iterations > max_iter:
				print( "Terminating due to too many iterations: ", max_iter )
				break
			
			print( "Starting iteration", iterations )
			
			## 1 Solve for ws.
			## 2 Solve for Ts.
			
			## 1
			if graph_laplacian == 'always' or ( graph_laplacian == 'once' and iterations == 1 ):
				ws = [ (1./len(neighs))*np.ones(len(neighs)) for neighs in neighbors ]
			else:
				ws_ssv_energy = [ laplacian.solve_for_w( Ts[i], Ts[ neighbors[i] ].T, return_energy = True ) for i in range( num_vertices ) ]
				ws = [ w for w, ssv, energy in ws_ssv_energy ]
				print( "E_local from ws point of view:", np.sum([ energy for w, ssv, energy in ws_ssv_energy ]) )
			
			## 2
			E_local = laplacian.quadratic_for_E_local( neighbors, ws, poses )
			E_local_val = laplacian.evaluate_E_local( E_local, Ts.T )
			print( "E_local from Ts point of view (before solving for Ts):", E_local_val )
			
			E_data_val = laplacian.evaluate_E_data( E_data, Ts.T )
			print( "E_data from Ts point of view (before solving for Ts):", E_data_val )
			
			f = E_data_val * E_data_weight + E_local_val * E_local_weight
			print( "=> E_total:", E_data_val + E_local_val )
			print( "=> E_total (weighted):", f )
			
			## Solve for T
			
			if strategy is None:
				Ts = laplacian.solve_for_T( E_data, E_local, poses, E_data_weight = E_data_weight, E_local_weight = E_local_weight ).T
			elif strategy == 'lerp':
				Ts = laplacian.solve_for_T( E_data, E_local, poses, E_data_weight = E_data_weight, E_local_weight = E_local_weight ).T
				## Lerp from the previous Ts
				Ts = Ts_prev + lerpval*(Ts - Ts_prev)
			elif strategy == 'lsq':
				Q_data, L_data, C_data = E_data
				## This could be done in advance.
				# Q_data = ( Q_data + scipy.sparse.diags( [ E_input_weight*(np.outer((1.0-lerpval),np.ones(12*P)).ravel(order='C')) ], [0] ) )
				Q_data = scipy.sparse.diags( [ E_input_weight*(np.outer((1.0-lerpval),np.ones(12*P)).ravel(order='C')) ], [0] )
				## This could not.
				# L_data = L_data - (2*E_input_weight) * ((1.0-lerpval) * qs_data).T.ravel(order='F')
				## This could.
				L_data = - (2*E_input_weight) * ((1.0-lerpval) * qs_data).T.ravel(order='F')
				# C_data = C_data + E_input_weight * np.dot( qs_data.ravel(order='F'), qs_data.ravel(order='F') )
				C_data = E_input_weight * np.dot( qs_data.ravel(order='F'), qs_data.ravel(order='F') )
				
				Ts = laplacian.solve_for_T( ( Q_data, L_data, C_data ), E_local, poses, E_data_weight = E_data_weight, E_local_weight = E_local_weight ).T
			
			## Plot
			if plot:
				Ts3D = mapper.project( Ts )
				ax.scatter( Ts3D.T[0], Ts3D.T[1], Ts3D.T[2] )
				plt.draw()
				plt.pause(0.05)
			
			## Get the function value after solving.
			E_local_val = laplacian.evaluate_E_local( E_local, Ts.T )
			print( "E_local from Ts point of view (after solving for Ts):", E_local_val )
			
			E_data_val = laplacian.evaluate_E_data( E_data, Ts.T )
			print( "E_data from Ts point of view (after solving for Ts):", E_data_val )
			
			f = E_data_val * E_data_weight + E_local_val * E_local_weight
			print( "=> E_total (after solving for Ts):", E_data_val + E_local_val )
			print( "=> E_total (weighted, after solving for Ts):", f )
			
			print( "Function value:", f )
			
			
			
			
			print( "Ts singular values:", np.linalg.svd( Ts, compute_uv = False ) )
			
			## If this is the first iteration, pretend that the old function value was
			## out of termination range.
			if f_prev is None: f_prev = f + 100*f_eps
			
			if f - f_prev > 0:
				print( "WARNING: Function value increased." )
			if abs( f_prev - f ) < f_eps:
				print( "Function change too small, terminating:", f_prev - f )
				converged = True
				break
			# x_change = abs( W_prev - W ).max()
			## To make xtol approximately match scipy's default gradient tolerance (gtol) for BFGS,
			## use norm() instead of the max change.
			x_change_norm = np.linalg.norm( Ts_prev - Ts )
			x_change_max = abs( Ts_prev - Ts ).max()
			print( "x change (norm):", x_change_norm )
			print( "x change (max):", x_change_max )
			x_change = x_change_max
			if x_change < x_eps:
				print( "Variables change too small, terminating:", x_change )
				converged = True
				break
			if f < f_zero_threshold:
				print( "Function below zero threshold, terminating." )
				converged = True
				break
			
			f_prev = f
			Ts_prev = Ts.copy()
	except KeyboardInterrupt:
		print( "Terminated by KeyboardInterrupt." )
	
	print( "Terminated after", iterations, "iterations." )
	
	from space_mapper import SpaceMapper
	pca = SpaceMapper.Uncorrellated_Space( Ts, dimension = H )
	p = pca.Xavg_
	B = pca.V_[:H-1].T
	
	if plot:
		while True:
			plt.pause(0.05)
	
	return converged, pack( p, B )

def optimize(P, H, all_R_mats, deformed_vs, x0):
	converged, x = optimize_approximated_quadratic(P, H, all_R_mats, deformed_vs, x0 )
	converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x )
	return converged, x
	

def per_vertex_transformation(x, P, row_mats, deformed_vs):
	
	import flat_intersection_biquadratic_gradients as biquadratic
	
	rev_vertex_transformations = []
	vertex_dists = []

	pt, B = unpack(x,P)
	num_underconstrained = 0
	for j, vs in enumerate(deformed_vs):
		vprime = vs.reshape((3*P,1))	
		vbar = row_mats[j]
		vB = np.dot( vbar, B )
		lh = np.dot( vB.T, vB )
		rh = -np.dot( vB.T, np.dot(vbar, pt) - vprime )
		## lh should be well-behaved. Its smallest singular value should not be zero.
		z = np.dot( np.linalg.pinv( lh ), rh )
		z = z.reshape(-1,1)
		
		sv = np.linalg.svd( lh, compute_uv = False )
		if sv[-1] < 1e-5:
			print( "Vertex", j, "has small singular values:", sv )
		
		transformation = pt + np.dot(B,z)
		rev_vertex_transformations.append( transformation )
		vertex_dists.append( np.linalg.norm( np.dot( vbar, transformation ) - vprime ) )
		
		z2, ssv = biquadratic.solve_for_z( np.hstack([ pt.reshape(-1,1), pt+B ]), vbar, vprime, return_energy = False, use_pseudoinverse = True )
		if ssv < 1e-5: continue
		transformation2 = np.dot( np.hstack([ pt.reshape(-1,1), pt+B ]), z2 )
# 		assert abs( transformation.squeeze() - transformation2.squeeze() ).max() < 1e-7
	
	return np.array( rev_vertex_transformations ).squeeze(), np.array( vertex_dists )
	

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser( description='Solve for transformation subspace.' )
	parser.add_argument( 'rest_pose', type=str, help='Rest pose (OBJ).')
	parser.add_argument( 'pose_folder', type=str, help='Folder containing deformed poses.')
	parser.add_argument('--handles', '-H', type=int, help='Number of handles.')
	parser.add_argument('--ground-truth', '-GT', type=str, help='Ground truth data path.')
	parser.add_argument('--recovery', '-R', type=float, help='Recovery test epsilon (default no recovery test).')
	parser.add_argument('--strategy', '-S', type=str, choices = ['function', 'gradient', 'hessian', 'mixed', 'grassmann', 'pinv', 'pinv+ssv:skip', 'pinv+ssv:weighted', 'ssv:skip', 'ssv:weighted'], help='Strategy: function, gradient (default), hessian, mixed, grassmann (for energy B only), pinv and ssv (for energy biquadratic only).')
	parser.add_argument('--energy', '-E', type=str, default='B', choices = ['B', 'cayley', 'B+cayley', 'B+B', 'cayley+cayley', 'biquadratic', 'biquadratic+B', 'biquadratic+handles', 'laplacian'], help='Energy: B (default), cayley, B+cayley, B+B, cayley+cayley, biquadratic, biquadratic+B, biquadratic+handles, laplacian.')
	## UPDATE: type=bool does not do what we think it does. bool("False") == True.
	##         For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	def str2bool(s): return {'true': True, 'false': False}[s.lower()]
	parser.add_argument('--solve-for-rest-pose', type=str2bool, default=False, help='Whether to solve for the rest pose (only affects "biquadratic" energy (default: False).')
	parser.add_argument('--error', type=str2bool, default=False, help='Whether to compute transformation error and vertex error compared with ground truth.')
	parser.add_argument('--zero', type=str2bool, default=False, help='Given ground truth, zero test.')
	parser.add_argument('--fancy-init', '-I', type=str, help='Valid points generated from local subspace intersection.')
	parser.add_argument('--fancy-init-errors', type=str, help='Errors for data generated from local subspace intersection.')
	parser.add_argument('--fancy-init-ssv', type=str, help='Smallest singular values for data generated from local subspace intersection.')
	parser.add_argument('--output', '-O', type=str, default="", help='output path.')
	parser.add_argument('--max-iter', type=int, help='Maximum number of iterations.')
	parser.add_argument('--f-eps', type=float, help='Function change epsilon (biquadratic).')
	parser.add_argument('--x-eps', type=float, help='Variable change epsilon (biquadratic).')
	parser.add_argument('--W-projection', type=str, choices = ['normalize', 'first', 'regularize_translation', 'regularize_identity', 'constrain_magnitude'], help='How to project W (biquadratic): normalize, first, regularize_translation, regularize_identity, constrain_magnitude.')
	parser.add_argument('--z-strategy', type=str, choices = ['positive'], help='How to solve for z (biquadratic): positive.')
	
	args = parser.parse_args()
	H = args.handles
	ground_truth_path = args.ground_truth
	recovery_test = args.recovery
	error_test = args.error
	zero_test = args.zero
	if error_test:	assert( ground_truth_path is not None and "Error test needs ground truth path." )
	if zero_test:	assert( ground_truth_path is not None and "Zero energy test or zero test need ground truth path." )
	
	fancy_init_path = args.fancy_init
	rest_mesh = TriMesh.FromOBJ_FileName( args.rest_pose )
	OBJ_name = os.path.splitext(os.path.basename(args.rest_pose))[0]
	print( "The name for the OBJ is:", OBJ_name )
	
	pose_paths = glob.glob(args.pose_folder + "/*.obj")
	pose_paths.sort()
	deformed_vs = np.array( [TriMesh.FromOBJ_FileName( path ).vs for path in pose_paths] )
	pose_name = os.path.basename( args.pose_folder )
	print( "The name for pose folder is:", pose_name )
	
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
				left_m	   [j*3+k, j*12+k*4:j*12+k*4+4] = pos_h
				
		all_rights[i] /= nm
		assert( np.allclose( np.dot(unit_left_m, unit_left_m.T), np.eye(3*P) ) )
		all_flats.append(unit_left_m)
		all_R_mats.append(left_m)
	all_flats = np.array( all_flats )
	all_R_mats = np.array( all_R_mats )
	
	print( "The rank of the stack of all pose row matrices is: ", np.linalg.matrix_rank( np.vstack( all_flats ) ) )
	
	start_time = time.time()
	if H is None:
		Hs = range(2, MAX_H)
	else:
		Hs = [H]
	
	x = None
	
	if ground_truth_path is not None:
		gt_bones, gt_vertices = zero_energy_test(ground_truth_path)
	
	for H in Hs:
	
		x0 = None
		## 0 energy test
		if zero_test:
			## Make it a fancy load, too.
			qs_data = gt_vertices

			pt = gt_bones[0]
			B = (gt_bones[1:] - pt).T
			x0 = pack( pt, B )
			## Recovery test
			if recovery_test is not None:
				np.random.seed(0)
				x0 += recovery_test*np.random.rand(12*P*H)
		
			print( "There are", len(gt_bones), "ground truth bones." )
			print( "If they are linearly independent, then the following has no zeros:", np.linalg.svd( B.T, compute_uv = False ) )

		else:
			print("#handles: ", H)
			if error_test:		np.random.seed(0)
			
			x0 = np.random.rand(H*12*P)
			
			if x is None:
				pt = np.zeros((3,4))
				pt[:,:3] = np.eye(3)
				pt = pt.ravel()
				for i in range(P):
					x0[12*i: 12*(i+1)] = pt
			else:
				x0[:12*P*(H-1)] = x
				
			pt, B = unpack( x0, P )
			for i in range(B.shape[1]):
				B[:,i] /= np.linalg.norm(B[:,i])
			
			if fancy_init_path is not None: 		
				qs_data = np.loadtxt(fancy_init_path)
				print( "# of good valid vertices: ", qs_data.shape[0] )
				from space_mapper import SpaceMapper
				pca = SpaceMapper.Uncorrellated_Space( qs_data, dimension = H )
				pt = pca.Xavg_
				B = pca.V_[:H-1].T
				
			x0 = pack( pt, B )
		
		if 3*P < B.shape[1]:
			print( "Warning: Not enough poses for the handles without pseudoinverse in the energy." )
		
		if args.energy == 'B':
			# converged, x = optimize(P, H, all_R_mats, deformed_vs, x0)
			converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy, max_iter = args.max_iter )
		elif args.energy == 'cayley':
			converged, x = optimize_nullspace_cayley( P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy )
		elif args.energy == 'B+cayley':
			converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy)
			converged, x = optimize_nullspace_cayley( P, H, all_R_mats, deformed_vs, x, strategy = args.strategy )
		elif args.energy == 'cayley+cayley':
			converged, x = optimize_nullspace_cayley( P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy )
			## This second one continues to improve.
			converged, x = optimize_nullspace_cayley( P, H, all_R_mats, deformed_vs, x, strategy = args.strategy )
		elif args.energy == 'B+B':
			converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy, max_iter = args.max_iter )
			## Without the following projection in and out of Grassmann space,
			## the next optimize_nullspace_directly() doesn't do anything.
			## (It terminates immediately.)
			p, B = unpack( x, P )
			import flat_intersection_cayley_grassmann_gradients as grassmann
			A = grassmann.A_from_non_Cayley_B( B )
			B = grassmann.B_from_Cayley_A( A, H )
			x = pack( p, B )
			converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x, strategy = args.strategy, max_iter = args.max_iter )
		elif args.energy == 'biquadratic':
			if args.solve_for_rest_pose:
				converged, x, new_all_R_mats = optimize_biquadratic( P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy, solve_for_rest_pose = args.solve_for_rest_pose, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, W_projection = args.W_projection, z_strategy = args.z_strategy )
			elif args.W_projection == 'first':
				assert args.fancy_init is not None
				assert args.fancy_init_errors is not None
				assert args.fancy_init_ssv is not None
				guess_data = np.loadtxt( args.fancy_init )
				guess_errors = np.loadtxt( args.fancy_init_errors )
				guess_ssv = np.loadtxt( args.fancy_init_ssv )
				converged, x = optimize_biquadratic( P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, W_projection = args.W_projection, z_strategy = args.z_strategy, guess_data = guess_data, guess_errors = guess_errors, guess_ssv = guess_ssv )
			else:
				converged, x = optimize_biquadratic( P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, W_projection = args.W_projection, z_strategy = args.z_strategy )
		elif args.energy == 'biquadratic+handles':
			converged, x = optimize_biquadratic( P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, W_projection = args.W_projection, z_strategy = args.z_strategy )
			p, B = unpack( x, P )
			B = B[:,:-1]
			x = pack( p, B )
			converged, x = optimize_biquadratic( P, H-1, all_R_mats, deformed_vs, x, strategy = args.strategy, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, W_projection = args.W_projection, z_strategy = args.z_strategy )
		elif args.energy == 'biquadratic+B':
			converged, x = optimize_biquadratic( P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, W_projection = args.W_projection, z_strategy = args.z_strategy )
			for i in range(10):
				print( "Now trying B for one iteration." )
				converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x, max_iter = 1)
				print( "Now biquadratic again." )
				converged, x = optimize_biquadratic( P, H, all_R_mats, deformed_vs, x, strategy = args.strategy, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, W_projection = args.W_projection, z_strategy = args.z_strategy )
		elif args.energy == 'laplacian':
			qs_errors = None
			if args.fancy_init_errors is not None: qs_errors = np.loadtxt(args.fancy_init_errors)
			qs_ssv = None
			if args.fancy_init_ssv is not None: qs_ssv = np.loadtxt(args.fancy_init_ssv)
			converged, x = optimize_laplacian( P, H, rest_mesh, deformed_vs, qs_data, qs_errors, qs_ssv, max_iter = args.max_iter, f_eps = args.f_eps, x_eps = args.x_eps, z_strategy = args.z_strategy )
		else:
			raise RuntimeError( "Unknown energy parameter: " + str(parser.energy) )
		
		if ground_truth_path is None and converged:
			print("Converged at handle #", H)
			break
	print( "Time for solving: ", time.time() - start_time )
	
	if ground_truth_path is not None and converged == False:
		print("Exceed Maximum #handles ", MAX_H)
	
	rev_vertex_trans, vertex_dists = per_vertex_transformation(x, P, all_R_mats, deformed_vs)	
	if error_test:
		transformation_error = abs( rev_vertex_trans - gt_vertices )
		print( "Largest, average and median transformation errors are: ", transformation_error.max(), transformation_error.mean(), np.median(transformation_error.ravel()) )
		print( "Largest, average and meidan vertex errors are: ", vertex_dists.max(), vertex_dists.mean(), np.median(vertex_dists) )


	output_folder = args.output
	if output_folder == "":
		curr_folder = os.path.abspath(".")
		assert os.path.basename(curr_folder) == "InverseSkinning"
		output_folder = os.path.join( "results", OBJ_name, + pose_name )
	
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	H = int(rev_vertex_trans.shape[1]/12)
	for i in range(H):
		per_pose_transformtion = rev_vertex_trans[:,i*12:(1+i)*12]
		per_pose_transformtion = per_pose_transformtion.reshape(-1,3,4)
		per_pose_transformtion = np.swapaxes(per_pose_transformtion, 1, 2).reshape(-1,12)
		output_path = os.path.join( output_folder, str(i+1) + ".DMAT" )
		format_loader.write_DMAT( output_path, per_pose_transformtion )
	
	