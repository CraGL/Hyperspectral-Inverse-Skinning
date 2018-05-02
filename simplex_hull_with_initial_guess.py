"""
Compute Convex hull from a set of OBJ poses. Affine transformations are col-major here.

Written by Songrun Liu
"""

from __future__ import print_function, division
from recordclass import recordclass

import os
import sys
import argparse
import time
import numpy as np
import scipy

import format_loader
from trimesh import TriMesh
from scipy.spatial import ConvexHull
import glob
from space_mapper import SpaceMapper

def simplex_volumn( pts ):
	'''
	pts should be N-by-N+1 dimensions
	'''
	assert( len( pts.shape ) == 2 )
	N = pts.shape[0]
	assert ( pts.shape[1] == N + 1 )
	Vinv = np.ones( ( N+1, N+1 ) )
	Vinv[:N, :] = pts
	
	## http://www.mathpages.com/home/kmath664/kmath664.htm
	invvol = abs( np.linalg.det( Vinv ) )
	
	return invvol


########################################
# CMD-line tool for getting filenames. #
########################################
if __name__ == '__main__':

	'''
	Uses ArgumentParser to parse the command line arguments.
	Input:
		parser - a precreated parser (If parser is None, creates a new parser)
	Outputs:
		Returns the arguments as a tuple in the following order:
			(in_mesh, Ts, Tmat)
	'''
		
	parser = argparse.ArgumentParser(description = "From per-vertex transformations to per-bone transformations. ", usage="%(prog)s path/to/input_model_folder")
	parser.add_argument("per_vertex_tranformation", type=str, help="Path to the folder containing input mesh files.")
	parser.add_argument( 'rest_pose', type=str, help='Rest pose (OBJ).')
	parser.add_argument( 'pose_folder', type=str, help='Folder containing deformed poses.')
	parser.add_argument('output', type=str, help='output path.')
	## UPDATE: type=bool does not do what we think it does. bool("False") == True.
	##         For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	def str2bool(s): return {'true': True, 'false': False}[s.lower()]
	parser.add_argument('--test', type=str2bool, default=False, help='testing mode.')
	parser.add_argument('--method', type=str, help='linear or quadratic solver: "lp" (default), "qp", "ipopt", "binary" or "scipy".')
	parser.add_argument('--linear-solver', '-L', type=str, help='Linear solver: "glpk" (default) or "mosek".')
	parser.add_argument('--max-iter', type=int, help='The maximum iterations for the solver.')
	parser.add_argument('--ground-truth', '-GT', type=str, help='Ground truth data path.')
	parser.add_argument('--robust-percentile', '-R', type=float, help='Fraction of outliers to discard. Default: 0.')
	parser.add_argument('--dimension', '-D', type=int, help='Dimension (number of handles minus one). Default: automatic.')
	parser.add_argument('--positive-weights', type=str2bool, default=False, help='If True, recovered weights must all be positive. If False, weights can be negative to better match vertices.')
	parser.add_argument('--min-weight', type=float, help='The minimum weight when solving.')
	parser.add_argument('--WPCA', type=str2bool, help='If True, uses weighted PCA instead of regular PCA. Requires')
	parser.add_argument('--transformation-errors', type=str, help='Errors for data generated from local subspace intersection.')
	parser.add_argument('--transformation-ssv', type=str, help='Smallest singular values for data generated from local subspace intersection.')
	## Only if the solver is still slow for big examples:
	parser.add_argument('--random-percent', type=float, help='If specified, compute with a random %% subset of the points. Default: off (equivalent to 100).')
	## This option is not recommended.
	parser.add_argument('--random-after-PCA', type=str2bool, default=False, help='Whether to take the random subset after computing PCA. Default: False.')
	parser.add_argument('--random-reps', type=int, default=1, help='How many times to repeat the random subsampling. Default: 1.')
	args = parser.parse_args()

	# Check that in_mesh exists
	if(not os.path.exists(args.per_vertex_tranformation)):
		parser.error("Path to per-vertex transformation txt does not exist.")
		
	base_dir = args.pose_folder
	if(not os.path.exists(args.pose_folder)):
		parser.error("Path to deformed pose folder does not exist.")
	mesh_paths = glob.glob(base_dir + "/*.obj")
	mesh_paths.sort()
	deformed_vs = np.array([ TriMesh.FromOBJ_FileName(mesh_path).vs for mesh_path in mesh_paths ])
	deformed_vs = np.swapaxes( deformed_vs, 0, 1 )
	
	if(not os.path.exists(args.rest_pose)):
		parser.error("Path to rest pose does not exist.")
	rest_mesh = TriMesh.FromOBJ_FileName(args.rest_pose)
	rest_vs = np.array(rest_mesh.vs)

	Ts = np.loadtxt(args.per_vertex_tranformation)
	## The following line fixes a bug in this code which assumed that the input was
	## in DMAT format, vertices-by-poses-by-four-by-three. The output from the initial
	## guess code was saving vertices-by-poses-by-four-by-three data (in row major order),
	## which is what flat_intersection.py takes as input. flat_intersection outputs
	## into DMAT format, which is appropriate for comparison with ground truth.
	## The following swapaxes() line does the appropriate swap, but it hasn't been tested.
	raise RuntimeError( "Test the following line." )
	Ts = np.swapaxes( Ts.reshape(-1,3,4), 1,2 ).reshape( Ts.shape[0], -1 )
	print( "# initial vertices: ", Ts.shape[0] )
	
	if args.WPCA is not None:
		Ts_errors = np.loadtxt( args.transformation_errors )
		Ts_ssv = np.loadtxt( args.transformation_ssv )
		Ts_weights = 1./(1e-5 + Ts_errors)
		Ts_weights[ Ts_ssv < 1e-8 ] = 0.
	
	if args.ground_truth is not None:
		handle_trans = glob.glob(args.ground_truth + "/*.Tmat")
		handle_trans.sort()
		Tmat = np.array([ format_loader.load_Tmat(transform_path) for transform_path in handle_trans ])
		Tmat = np.swapaxes(Tmat,0,1)
		Tmat = Tmat.reshape(Tmat.shape[0], Tmat.shape[1]*Tmat.shape[2])
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	## For backwards compatibility with args.test:
	if args.test:
		args.random_percent = 10
	
	# np.random.seed(0)
	
	import mves2
	
	np.set_printoptions(precision=4, suppress=True)
	startTime = time.time()
	
	all_Ts = Ts.copy()
	if args.WPCA is not None: all_Ts_weights = Ts_weights.copy()
	## results stores: volume, solution, iter_num
	results = []
	for random_rep in range( args.random_reps ):
		Ts = all_Ts.copy()
		if args.WPCA is not None: Ts_weights = all_Ts_weights.copy()
		
		if args.random_percent is not None and not args.random_after_PCA:
			keep_N = np.clip( int( ( args.random_percent * len(Ts) )/100. + 0.5 ), 0, len( Ts ) )
			## For some reason built-in numpy.random function produce worse results.
			## This must be superstition!
			# Ts = np.random.permutation( Ts )[:keep_N]
			# np.random.shuffle( Ts )
			# Ts = Ts[:keep_N]
			import random
			indices = np.arange( len( Ts ) )
			random.shuffle( indices )
			indices = indices[:keep_N]
			Ts = Ts[indices]
			if args.WPCA is not None: Ts_weights = Ts_weights[indices]
			
			print( "Keeping %s out of %s points (before PCA)." % ( len( Ts ), len( all_Ts ) ) )
		
		if args.WPCA is not None:
			## This code requires wpca (https://github.com/jakevdp/wpca):
			### pip install wpca
			### pip install scikit-learn
			
			# from wpca import EMPCA
			from wpca import WPCA
			class WeightedSpaceMapper( object ):
				def __init__( self, data, weights, dimension = None ):
					assert dimension is not None
					self.mapper = WPCA( n_components = dimension ).fit( data, weights = np.repeat( weights.reshape(-1,1), data.shape[1], axis = 1 ) )
					print( "WPCA amount of variance explained by each of the selected components:", self.mapper.explained_variance_ )
					print( "WPCA Percentage of variance explained by each of the selected components:", self.mapper.explained_variance_ratio_ )
				def project( self, points ):
					return self.mapper.transform( points )
				def unproject( self, low_dim_points ):
					return self.mapper.inverse_transform( low_dim_points )
			WPCA_startTime = time.time()
			Ts_mapper = WeightedSpaceMapper( Ts, Ts_weights, dimension = args.dimension )
			WPCA_running_time = time.time() - WPCA_startTime
			print("Weighted PCA took: %.2f seconds" % WPCA_running_time)
		else:
			Ts_mapper = SpaceMapper.Uncorrellated_Space( Ts, dimension = args.dimension )
		uncorrelated = Ts_mapper.project( Ts )
		print( "uncorrelated data shape" )
		print( uncorrelated.shape )
	
		if args.random_percent is not None and args.random_after_PCA:
			uncorrelated_all = uncorrelated
			keep_N = np.clip( int( ( args.random_percent * len(uncorrelated) )/100. + 0.5 ), 0, len( uncorrelated ) )
			uncorrelated = np.random.permutation( uncorrelated )[:keep_N]
			print( "Keeping %s out of %s points (after PCA)." % ( len( uncorrelated ), len( uncorrelated_all ) ) )
	
		# import scipy.io
		# scipy.io.savemat( 'MVES_input.mat', mdict={'M': uncorrelated})
		# print( "Saved input points to MVES in MATLAB format as:", 'MVES_input.mat' )
	 
		## Compute minimum-volume enclosing simplex
		solution, weights, iter_num = mves2.MVES( uncorrelated, method=args.method, linear_solver = args.linear_solver, max_iter = args.max_iter, min_weight = args.min_weight )
		volume = abs( np.linalg.det( solution ) )
		
		print( "solution" )
		print( solution )
	
		
		print( "solve weights from initial guess finished" )
		
		## Cheap robustness; discard the % of data which ended up with the smallest weights.
		## Outliers will always be on faces, so they will have a 0 weight for some vertex.
		## Discard some of them.
		if args.robust_percentile is not None:
			# argsorted = weights.argsort(axis=0).ravel()
			argsorted = np.argsort((weights < 1e-8).sum(1))[::-1]
			num_rows_to_discard = int( (args.robust_percentile*len(weights))/100. + 0.5 )
			print( "Deleting", num_rows_to_discard, "rows with the smallest weights." )
			rows_to_discard = argsorted[ :num_rows_to_discard ]
			uncorrelated_robust = np.delete( uncorrelated, rows_to_discard, axis = 0 )
			print( "Re-running MVES" )
			solution, weights_robust, iter_num = mves2.MVES( uncorrelated_robust, method = args.method, linear_solver = args.linear_solver,
				max_iter = args.max_iter, min_weight = args.min_weight )
			weights = weights_robust
			volume = abs( np.linalg.det( solution ) )
			print( "robust solution" )
			print( solution )
		
		results.append( ( volume, solution, iter_num, Ts_mapper ) )
	
	## TODO Q: min() or max()? min() is good for outliers. max() is better if we want
	##         to avoid losing parts (larger error when restricted to positive weights).
	volume, solution, iter_num, Ts_mapper = max( results )
	print( "=> Best simplex found with volume:", volume )
	
	running_time = time.time() - startTime
	print("\nOptimization costs: %.2f seconds" %running_time)
	print( "solution simplex volumn: ", simplex_volumn( solution[:-1] ).round(4) )
	
	recovered = Ts_mapper.unproject( solution[:-1].T )
	print( 'recovered', recovered.shape )
	print( recovered.round(3) )

	## Because we load a potentially incomplete initial guess, we need to recover
	## the weights for all points manually. We could do this with PCA projection
	## and multiplying the inverse of solution. Or we could re-use code from
	## one of our solvers.
	import flat_intersection_biquadratic_gradients as biquadratic
	N,B = rest_vs.shape[0], recovered.shape[0]
	P = int(recovered.shape[1]//12)
	weights = np.zeros((N,B))
	if args.positive_weights:
		solve_for_z_kwargs = {'return_energy': False, 'use_pseudoinverse': False, 'strategy': 'positive'}
	else:
		solve_for_z_kwargs = {'return_energy': False, 'use_pseudoinverse': True}
	for i in range(N):
		weights[i], ssv = biquadratic.solve_for_z(
			recovered.T,
			np.append( rest_vs[i], [1] ).reshape(1,-1),
			deformed_vs[i].ravel(),
			**solve_for_z_kwargs
			)
		# transformation = np.dot( recovered.T, weights )
	
	print( "Minimum weight:", weights.min() )
	print( "Number of points with negative weights (< -1e-5):", ( weights < -1e-5 ).any(axis=1).sum() )
	print( "Number of points with negative weights (< -0.1):", ( weights < -0.1 ).any(axis=1).sum() )
	print( "Number of points with negative weights (< -0.5):", ( weights < -0.5 ).any(axis=1).sum() )
	print( "Number of points with negative weights (< -1):", ( weights < -1 ).any(axis=1).sum() )
	
	'''
	N,B = rest_vs.shape[0], recovered.shape[0]
	P = int(recovered.shape[1]/12)
	weights = np.zeros((N,B))
	for i in range(N):
		v = np.append(rest_vs[i], 1.)
		mat_v = np.zeros((3,12))	
		mat_v[0,:4] = mat_v[1,4:8] = mat_v[2,8:12] = v
		
		## pack the deformed position using one bone's transformation for all the poses
		lh = np.zeros((3*P,B))
		for k in range(B):
			data_per_bone = recovered[k].reshape(P,12).T
			for j in range(P):
				lh[j*3:(j+1)*3,k] = np.dot(mat_v, data_per_bone[:,j])
		
		rh = deformed_vs[i].ravel()
		x = np.linalg.solve(lh,rh)
		weights[i] = x	
	'''
	
	output_path = args.output
	print( "Saving recovered results to:", output_path )
	format_loader.write_result(output_path, recovered, weights, iter_num, running_time, col_major=False)
	
	def check_recovered( recovered, ground ):
		flags = np.zeros( len(Tmat), dtype = bool )
		dists = np.zeros( len(Tmat) )
		for i, ht in enumerate( recovered ):
			min_dist = np.linalg.norm( ht - ground[0] )
			for j, gt in enumerate( ground ): 
				min_dist = min( min_dist, np.linalg.norm( ht - gt ) )
				if np.allclose(ht, gt, rtol=1e-1, atol=1e-2):
					flags[i] = True
					ground = np.delete(ground, j, 0)
					break
			dists[i] = min_dist
		return flags, ground, dists
		
	if args.ground_truth is not None:
		print( "ground truth simplex volumn: ", simplex_volumn( Ts_mapper.project( Tmat ).T ).round(4) )
		status, remains, dists = check_recovered( recovered, Tmat )
		print( "recovered deviation: ", dists )
		print( "Average recovered deviation: ", dists.mean().round(4) )
		if( all( status ) ):	print( "match ground truth" )
		else:
			print( "#unmatched: ", np.nonzero( ~status ) )
			print( "Unmatched recovery:" )
			print( recovered[ np.nonzero( ~status ) ].round(4) )
			print( "Unmatched ground truth: " )
			print( remains.round(4) )
		
	

	

