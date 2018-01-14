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
	parser.add_argument('--test', type=bool, default=False, help='testing mode.')
	parser.add_argument('--method', type=str, help='linear or quadratic solver: "lp" (default), "qp", "ipopt", "binary" or "scipy".')
	parser.add_argument('--linear-solver', '-L', type=str, help='Linear solver: "glpk" (default) or "mosek".')
	parser.add_argument('--ground-truth', '-GT', type=str, help='Ground truth data path.')
	parser.add_argument('--robust-percentile', '-R', type=float, help='Fraction of outliers to discard. Default: 0.')
	parser.add_argument('--dimension', '-D', type=int, help='Dimension (number of handles minus one). Default: automatic.')
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
	print( "# initial vertices: ", Ts.shape[0] )
	
	if( args.test ):
		import random
		random.shuffle( Ts )
		Ts = Ts[:int(Ts.shape[0]*0.1)]
	
	if args.ground_truth is not None:
		handle_trans = glob.glob(args.ground_truth + "/*.Tmat")
		handle_trans.sort()
		Tmat = np.array([ format_loader.load_Tmat(transform_path) for transform_path in handle_trans ])
		Tmat = np.swapaxes(Tmat,0,1)
		Tmat = Tmat.reshape(Tmat.shape[0], Tmat.shape[1]*Tmat.shape[2])
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	Ts_mapper = SpaceMapper.Uncorrellated_Space( Ts, dimension = args.dimension )
	uncorrelated = Ts_mapper.project( Ts )
	print( "uncorrelated data shape" )
	print( uncorrelated.shape )

	startTime = time.time()
	np.set_printoptions(precision=4, suppress=True)
 
	## Compute minimum-volume enclosing simplex
	import mves2
	solution, weights, iter_num = mves2.MVES( uncorrelated, method=args.method, linear_solver = args.linear_solver )
	
	print( "solution" )
	print( solution )

		
	print( "solve weights from initial guess finished" )
	
	## Cheap robustness; discard the % of data which ended up with the smallest weights.
	## Outliers will always have 
	if args.robust_percentile is not None:
		argsorted = weights.argsort(axis=0)
		num_rows_to_discard = int( args.robust_percentile*len(weights) )
		print( "Deleting", num_rows_to_discard, "rows with the smallest weights." )
		rows_to_discard = argsorted[ :num_rows_to_discard ].ravel()
		uncorrelated_robust = np.delete( uncorrelated, rows_to_discard, axis = 0 )
		print( "Re-running MVES" )
		solution, weights_robust, iter_num = mves2.MVES( uncorrelated_robust )
		weights = np.dot( np.linalg.inv( solution ), np.concatenate( ( uncorrelated.T, np.ones((1,uncorrelated.shape[0])) ), axis=0 ) ).T
		print( "robust solution" )
		print( solution )
	
	running_time = time.time() - startTime
	print("\nOptimization costs: %.2f seconds" %running_time)
	print( "solution simplex volumn: ", simplex_volumn( solution[:-1] ).round(4) )
	
	recovered = Ts_mapper.unproject( solution[:-1].T )
	print( 'recovered', recovered.shape )
	print( recovered.round(3) )

	import flat_intersection_biquadratic_gradients as biquadratic
	N,B = rest_vs.shape[0], recovered.shape[0]
	P = int(recovered.shape[1]//12)
	weights = np.zeros((N,B))
	for i in range(N):
		weights[i], ssv = biquadratic.solve_for_z(
			recovered.T,
			np.kron( np.identity(P*3), np.append( rest_vs[i], [1] ).reshape(1,-1) ),
			deformed_vs[i].ravel(),
			return_energy = False, use_pseudoinverse = True
			)
		# transformation = np.dot( recovered.T, weights )
	
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
	format_loader.write_result(output_path, recovered.round(6), weights.round(6), iter_num, running_time, col_major=False)
	
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
		
	

	

