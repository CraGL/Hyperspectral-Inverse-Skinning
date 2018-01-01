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
	
	gt_W_path = "models/cheburashka/cheburashka.DMAT"
	gt_W = np.array( format_loader.load_DMAT(gt_W_path) ).T
	
	return gt_bones, gt_vertices, gt_W
	
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
	return normalization

def optimize_nullspace_directly(P, H, row_mats, deformed_vs, x0, strategy = None):

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
		solution = scipy.optimize.minimize( f_point_distance_sum, x0, constraints = constraints, method = 'L-BFGS-B', callback = show_progress, options={'maxiter':10, 'disp':True} )
	elif strategy == 'gradient':
		## With gradients:
		## check_grad() is too slow to always run.
		# grad_err = scipy.optimize.check_grad( lambda x: f_point_distance_sum_and_gradient(x)[0], lambda x: f_point_distance_sum_and_gradient(x)[1], x0 )
		# print( "scipy.optimize.check_grad() error:", grad_err )
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, method = 'L-BFGS-B', callback = show_progress, options={'disp':True} )
		solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, callback = show_progress, options={'disp':True} )
	elif strategy == 'hessian':
		## Use the Hessian:
		# solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, hess = f_hess, method = 'Newton-CG', callback = show_progress, options={'disp':True} )
		solution = scipy.optimize.minimize( f_point_distance_sum_and_gradient, x0, jac = True, hess = f_hess, method = 'Newton-CG', callback = show_progress, options={'disp':True} )
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
			print( "Terminating due to too many iterations." )
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

def optimize_biquadratic(P, H, row_mats, deformed_vs, x0, f_eps = None, x_eps = None, max_iter = None):

	## To make function values comparable, we need to normalize.
	xyzs = np.asarray([ row_mat[0,:3] for row_mat in row_mats ])
	diag = xyzs.max( axis = 0 ) - xyzs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	normalization = 1./( len(row_mats) * diag )
	
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
		f_eps = 1e-10
	if x_eps is None:
		x_eps = 1e-10
	if max_iter is None:
		max_iter = 9999
	
	import flat_intersection_biquadratic_gradients as biquadratic
	
	f_prev = None
	W_prev = unpack_W( x0.copy(), P )
	iterations = 0
	converged = False
	while( True ):
		iterations += 1
		if iterations > max_iter:
			print( "Terminating due to too many iterations." )
			break
		
		print( "Starting iteration", iterations )
		
		## 1 Find the optimal z.
		## 2 Accumulate the linear matrix equation for W.
		## 3 Solve for the new W.
		
		f = 0
		As = []
		Bs = []
		Ys = []
		
		for i, vs in enumerate(deformed_vs):
			vprime = vs.reshape((3*P,1))
			vbar = row_mats[i]
			
			## 1
			z, fi = biquadratic.solve_for_z( W_prev, vbar, vprime, return_energy = True )
			
			## 2
			A, B, Y = linear_matrix_equation_for_W( vbar, vprime, z )
			As.append( A )
			Bs.append( B )
			Ys.append( Y )
			
			f += fi
		
		## 3
		W = biquadratic.solve_for_W( As, Bs, Ys )
		f *= normalization
		
		## If this is the first iteration, pretend that the old function value was
		## out of termination range.
		if f_prev is None: f_prev = f + 100*f_eps
		
		if f - f_prev > 0:
			print( "WARNING: Function value increased." )
		if abs( f_prev - f ) < f_eps:
			print( "Function change too small, terminating:", f_prev - f )
			converged = True
			break
		x_change = abs( W_prev - W ).max()
		if W_change < x_eps:
			print( "Variables change too small, terminating:", x_change )
			converged = True
			break
		
		f_prev = f
		W_prev = W.copy()
	
	print( "Terminated after", iterations, "iterations." )
	
	return converged, pack_W( W )

def optimize(P, H, all_R_mats, deformed_vs, x0):
	converged, x = optimize_approximated_quadratic(P, H, all_R_mats, deformed_vs, x0 )
	converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x )
	return converged, x
	

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser( description='Solve for transformation subspace.' )
	parser.add_argument( 'rest_pose', type=str, help='Rest pose (OBJ).')
	parser.add_argument( 'deformed_vs', type=str, help='Deformed vertices.')
	parser.add_argument('--handles', '--H', type=int, help='Number of handles.')
	parser.add_argument('--ground-truth', '--GT', type=str, help='Ground truth data path.')
	parser.add_argument('--recovery', '--R', type=float, help='Recovery test epsilon (default no recovery test).')
	parser.add_argument('--strategy', '--S', type=str, choices = ['function', 'gradient', 'hessian', 'mixed', 'grassmann'], help='Strategy: function, gradient (default), hessian, mixed, grassmann (for energy B only).')
	parser.add_argument('--energy', '--E', type=str, default='B', choices = ['B', 'cayley', 'B+cayley', 'B+B', 'cayley+cayley', 'biquadratic'], help='Energy: B (default), cayley, B+cayley, B+B, cayley+cayley.')
	
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
				left_m	   [j*3+k, j*12+k*4:j*12+k*4+4] = pos_h
				
		all_rights[i] /= nm
		assert( np.allclose( np.dot(unit_left_m, unit_left_m.T), np.eye(3*P) ) )
		all_flats.append(unit_left_m)
		all_R_mats.append(left_m)
	all_flats = np.array( all_flats )
	all_R_mats = np.array( all_R_mats )
	
	print( "The rank of the stack of all pose row matrices is: ", np.linalg.matrix_rank( np.vstack( all_flats ) ) )
	
	## find flat intersection
#	start_time = time.time()
#	intersect = all_flats[0]
#	for flat in all_flats[1:]:
#		intersect = intersect_flat( intersect, flat )
#	print( "Time for intersect all flats iteratively: ", time.time() - start_time )

	seed = []
	for i in range( N ):
		for j in range( P ):
			dist = (deformed_vs[i,j] - rest_mesh.vs[i])[:,np.newaxis]
			translation = np.hstack((np.eye(3), dist)).reshape(-1)
			seed.append( translation )
	seed = np.array( seed ).reshape( N, -1 )	
	assert( np.max( abs( np.dot(all_flats[0], seed[0].reshape(-1,1)) - all_rights[0].reshape(-1,1) ) ) < 1e-10 )
	
#	optimize_flat_intersection(P, all_flats, seed)
	if H is None:
		Hs = range(2, MAX_H)
	else:
		Hs = [H]
	
	x = None
	for H in Hs:
	
		x0 = None
		## 0 energy test
		if ground_truth_path is not None:
			gt_bones, gt_vertices, gt_W = zero_energy_test(ground_truth_path)

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
			x0 = pack( pt, B )
		
		if 3*P < B.shape[1]:
			print( "Warning: Not enough poses for the handles without pseudoinverse in the energy." )
		
		if args.energy == 'B':
			# converged, x = optimize(P, H, all_R_mats, deformed_vs, x0)
			converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy)
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
			converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x0, strategy = args.strategy)
			## Without the following projection in and out of Grassmann space,
			## the next optimize_nullspace_directly() doesn't do anything.
			## (It terminates immediately.)
			p, B = unpack( x, P )
			import flat_intersection_cayley_grassmann_gradients as grassmann
			A = grassmann.A_from_non_Cayley_B( B )
			B = grassmann.B_from_Cayley_A( A, H )
			x = pack( p, B )
			converged, x = optimize_nullspace_directly(P, H, all_R_mats, deformed_vs, x, strategy = args.strategy)
		elif args.energy == 'biquadratic':
			converged, x = optimize_biquadratic( P, H, all_R_mats, deformed_vs, x0 )
		else:
			raise RuntimeError( "Unknown energy parameter: " + str(parser.energy) )
		
		if ground_truth_path is None and converged:
			print("Converged at handle #", H)
			break
	
	if ground_truth_path is not None:
		print("Exceed Maximum #handles ", MAX_H)

