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
import numpy
import numpy as np
import scipy

import format_loader
from trimesh import TriMesh
from scipy.spatial import ConvexHull
import glob
from space_mapper import SpaceMapper

def uniquify_points_and_return_input_index_to_unique_index_map( pts, digits = 0 ):
	'''
	Given a sequence of N points 'pts',
	and an optional 'digits' indicating how many decimal places of accuracy (default: 0)
	returns two items:
	   a sequence of all the unique elements in 'pts'
	   and
	   a list of length N where the i-th item in the list tells you where
	   pts[i] can be found in the unique elements.
	
	From: https://github.com/songrun/VectorSkinning/blob/master/src/weights_computer.py
	'''
	
	from collections import OrderedDict
	unique_pts = OrderedDict()
	pts_map = []
	## Add rounded points to a dictionary and set the key to
	## ( the index into the ordered dictionary, the non-rounded point )
	for i, ( pt, rounded_pt ) in enumerate( zip( pts, map( tuple, numpy.asarray( pts ).round( digits ) ) ) ):
		index = unique_pts.setdefault( rounded_pt, ( len( unique_pts ), pt ) )[0]
		## For fancier schemes:
		# index = unique_pts.setdefault( rounded_pt, ( len( unique_pts ), [] ) )[0]
		# unique_pts[ rounded_pt ][1].append( pt )
		pts_map.append( index )
	
	## Return the original resolution points.
	## The average of all points that round:
	# return [ tuple( average( pt, axis = 0 ) ) for i, pt in unique_pts.itervalues() ], pts_map
	## The closest point to the rounded point:
	# return [ tuple( pt[ abs( asarray( pt ).round( digits ) - asarray( pt ) ).sum(axis=1).argmin() ] ) for i, pt in unique_pts.itervalues() ], pts_map
	## Simplest, the first rounded point:
	return [ tuple( pt ) for i, pt in unique_pts.itervalues() ], pts_map

def simplex_volumn( pts ):
	'''
	pts should be N-by-N+1 dimensions
	'''
	assert( len( pts.shape ) == 2 )
	N = pts.shape[0]
	assert ( pts.shape[1] == N + 1 )
	Vinv = numpy.ones( ( N+1, N+1 ) )
	Vinv[:N, :] = pts
	
	## http://www.mathpages.com/home/kmath664/kmath664.htm
	invvol = abs( numpy.linalg.det( Vinv ) )
	
	return invvol


def divide_mesh_into_small_sets( mesh, Ts, MAX_DIMENSION = 5 ):
	'''
	divide mesh into subset whose dimension is smaller
	'''
	assert( len(mesh.vs) == Ts.shape[0] )
	N = len(mesh.vs)
	visited = numpy.array([False] * N )
	small_sets = []
	
	while True:
		# Find all unvisited vertices
		pool = numpy.where( ~visited )[0]
		## stop if all the vertices have been visited
		print( "# unvisited: ", len(pool) )
		if len(pool) == 0:	break
		# randomly select one unvisited vertex
		unvisited_index = pool[0] # numpy.random.choice( pool )
		
		current_vertex_set = [ unvisited_index ]
		expansion_set = [ unvisited_index ]
		visited[ unvisited_index ] = True
			
		while True:
			# Expand with breadth first search
			# expansion_set	 <- unvisited neighbors of ( current_vertex_set )
			prev_expansion_set = expansion_set
			expansion_set = []
			for i in prev_expansion_set:
				neighbor_vertices = mesh.vertex_vertex_neighbors(i)
				for j in neighbor_vertices:
					if visited[ j ] == False:
						visited[ j ] = True
						expansion_set.append( j )
	
			# PCA( current_vertex_set union expansion_set ) dimension > max_dimension: break
			X = numpy.take( Ts, current_vertex_set + expansion_set, axis=0 )
			if SpaceMapper.PCA_Dimension( X ) > MAX_DIMENSION:
				for i in expansion_set:
					visited[i] = False
				break
			
			# current_vertex_set <- current_vertex_set union expansion_set
			current_vertex_set = current_vertex_set + expansion_set
#			print( "# unvisited: ", N - sum(visited) )
			if len(expansion_set) == 0:
				break

		# Add the convex hull of the current vertex set to the list of small_sets
		current_points = numpy.take( Ts, current_vertex_set + expansion_set, axis=0 )
		small_sets.append( current_points )
		
# 		if( len( current_vertex_set + expansion_set ) > Ts.shape[1] ):
# 			current_points = numpy.take( Ts, current_vertex_set + expansion_set, axis=0 )
# 			hull = ConvexHull( current_points )
# 			hull_vertices = numpy.take( hull.points, hull.vertices, axis=0 )
# 			small_sets.append( hull_vertices )
		
# 		if( len( current_vertex_set + expansion_set ) > MAX_DIMENSION+1 ):
# 			current_points = numpy.take( Ts, current_vertex_set + expansion_set, axis=0 )
# 			mapper = SpaceMapper.Uncorrellated_Space( current_points, False )
# 			hull = ConvexHull( mapper.project( current_points ) )
# 			hull_vertices = numpy.take( hull.points, hull.vertices, axis=0 )
# 			small_sets.append( mapper.unproject( hull_vertices ) )

	return numpy.array( small_sets )
			

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
	parser.add_argument('--method', type=str, help='linear or quadratic solver: "lp" (default), "qp", "ipopt", "binary" or "scipy".')
	parser.add_argument('--linear-solver', '-L', type=str, help='Linear solver: "glpk" (default) or "mosek".')
	parser.add_argument('--max-iter', type=int, help='The maximum iterations for the solver.')
	parser.add_argument('--ground-truth', '-GT', type=str, help='Ground truth data path.')
	parser.add_argument('--robust-percentile', '-R', type=float, help='Fraction of outliers to discard. Default: 0.')
	parser.add_argument('--dimension', '-D', type=int, help='Dimension (number of handles minus one). Default: automatic.')
	parser.add_argument('--output', '-O', type=str, help="output path")
	## Only if the solver is still slow for big examples:
	parser.add_argument('--random-percent', type=float, help='If specified, compute with a random % subset of the points. Default: off (equivalent to 100).')
	## UPDATE: type=bool does not do what we think it does. bool("False") == True.
	##         For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	def str2bool(s): return {'true': True, 'false': False}[s.lower()]
	## This option is not recommended.
	parser.add_argument('--random-after-PCA', type=str2bool, default=False, help='Whether to take the random subset after computing PCA. Default: False.')
	parser.add_argument('--random-reps', type=int, default=1, help='How many times to repeat the random subsampling. Default: 1.')
	args = parser.parse_args()

	# Check that in_mesh exists
	if(not os.path.exists(args.per_vertex_tranformation)):
		parser.error("Path to per-vertex transformation folder does not exist.")

	per_vertex_folder = args.per_vertex_tranformation
	in_transformations = glob.glob(per_vertex_folder + "/*.DMAT")
	in_transformations.sort()
	Ts = numpy.array([ format_loader.load_DMAT(transform_path).T for transform_path in in_transformations ])	
	num_poses = Ts.shape[0]
	num_verts = Ts.shape[1]
	Ts = numpy.swapaxes(Ts,0,1)
	Ts = Ts.reshape(num_verts, -1)
	
	if args.ground_truth is not None:
		handle_trans = glob.glob(args.ground_truth + "/*.Tmat")
		handle_trans.sort()
		Tmat = numpy.array([ format_loader.load_Tmat(transform_path) for transform_path in handle_trans ])
		Tmat = numpy.swapaxes(Tmat,0,1)
		Tmat = Tmat.reshape(Tmat.shape[0], Tmat.shape[1]*Tmat.shape[2])
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	
#	print( 'Ts.shape:', Ts.shape )
#	Ts_unique, unique_to_original_map = uniquify_points_and_return_input_index_to_unique_index_map( Ts, digits = 5 )
#	Ts_unique = numpy.asarray( Ts_unique )
#	print( 'Unique Ts.shape:', Ts_unique.shape )
	
	numpy.set_printoptions(precision=4, suppress=True)
	import os,sys
	import mves2
	
	startTime = time.time()
	
	all_Ts = Ts.copy()
	## results stores: volume, solution, iter_num
	results = []
	
	for random_rep in range( args.random_reps ):
		Ts = all_Ts.copy()
		
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
			
			print( "Keeping %s out of %s points (before PCA)." % ( len( Ts ), len( all_Ts ) ) )
		
		Ts_mapper = SpaceMapper.Uncorrellated_Space( Ts, dimension = args.dimension )
		uncorrelated = Ts_mapper.project( Ts )
		print( "uncorrelated data shape" )
		print( uncorrelated.shape )
	# 	Ts_sets = divide_mesh_into_small_sets(meshes[0], uncorrelated, 2)
	# 	N = len( uncorrelated )
	# 	Ts_sets = [ uncorrelated[:N/4], uncorrelated[N/4:N/2], uncorrelated[N/2:N*3/4], uncorrelated[N*3/4:] ]
	# 	print( "# small sets: ", len( Ts_sets ) )
		
		if args.random_percent is not None and args.random_after_PCA:
			uncorrelated_all = uncorrelated
			keep_N = np.clip( int( ( args.random_percent * len(uncorrelated) )/100. + 0.5 ), 0, len( uncorrelated ) )
			uncorrelated = np.random.permutation( uncorrelated )[:keep_N]
			print( "Keeping %s out of %s points (after PCA)." % ( len( uncorrelated ), len( uncorrelated_all ) ) )
		
	#	dirs = sys.argv[-1].split(os.path.sep)
	#	assert( len(dirs) > 2 )
	#	path = "benchmarks/" + dirs[-2] + "-" + dirs[-1] + ".csv"
	#	print("path: ", path)
	#	numpy.savetxt(path, uncorrelated, fmt="%1.6f", delimiter=",")
	#	outpath = "benchmarks/" + dirs[-2] + "-" + dirs[-1] + ".mat"
	#	import scipy.io
	#	scipy.io.savemat( outpath, { 'X': uncorrelated } )
		
		## plot uncorrelated data
	#	import matplotlib.pyplot as plt
	#	from mpl_toolkits.mplot3d import Axes3D
	#	
	#	plt3d = plt.figure().gca(projection='3d')
	#	plt3d.scatter(uncorrelated[:,0] , uncorrelated[:,1] , uncorrelated[:,2],  color='green')
	#	plt.show()
	 
		## Compute minimum-volume enclosing simplex
		solution, weights, iter_num = mves2.MVES( uncorrelated, method = args.method, linear_solver = args.linear_solver, max_iter = args.max_iter )
		volume = abs( np.linalg.det( solution ) )
		
		'''
		## Option 1: Run MVES on the union of convex hull vertices:
		good_verts = []
		for verts in Ts_sets:
			mapper = SpaceMapper.Uncorrellated_Space( verts, False )
			projected_verts = mapper.project( verts )
			if len( projected_verts ) > mapper.stop_s+1:
				hull = ConvexHull( projected_verts )
				projected_hulls = numpy.take( hull.points, hull.vertices, axis=0 )
				if good_verts == []: 
					good_verts = mapper.unproject( projected_hulls )
				else:
					good_verts = numpy.concatenate( (good_verts, mapper.unproject( projected_hulls )), axis=0 )
			else:
				if good_verts == []:
					good_verts = projected_verts
				else:
					good_verts = numpy.concatenate( (good_verts, mapper.unproject( projected_verts )), axis=0 )
		print( "option 1: # vertices is ", len(good_verts) )
		solution = mves2.MVES( good_verts )
		'''
		
		## Option 2: Run MVES on the small sets
	# 	good_verts = []
	# 	for verts in Ts_sets:
	# 		mapper = SpaceMapper.Uncorrellated_Space( verts, False )
	# 		projected_verts = mapper.project( verts )
	# 		if len( projected_verts ) > mapper.stop_s+1:
	# 			hull = ConvexHull( projected_verts )
	# 			projected_hull = numpy.take( hull.points, hull.vertices, axis=0 )
	# 			restored_hull = mapper.unproject( mves2.MVES( projected_hull ).x[:-1].T )
	# 			if good_verts == []:
	# 				good_verts =  restored_hull
	# 			else:
	# 				good_verts = numpy.concatenate( (good_verts, restored_hull), axis=0 )
	# 		else:
	# 			restored_verts = mapper.unproject( mves2.MVES( projected_verts ).x[:-1].T )
	# 			if good_verts == []:
	# 				good_verts = mapper.unproject( restored_verts )
	# 			else:
	# 				good_verts = numpy.concatenate( (good_verts, restored_verts), axis=0 )
	# 	
	# 	solution = mves2.MVES( good_verts )
		
		print( "solution" )
		print( solution )
		
		## Cheap robustness; discard the % of data which ended up with the smallest weights.
		## Outliers will always be on faces, so they will have a 0 weight for some vertex.
		## Discard some of them.
		if args.robust_percentile is not None:
			# argsorted = weights.argsort(axis=0).ravel()
			argsorted = np.argsort((weights < 1e-8).sum(1))[::-1]
			num_rows_to_discard = int( (args.robust_percentile*len(weights))/100. + 0.5 )
			print( "Deleting", num_rows_to_discard, "rows with the smallest weights." )
			rows_to_discard = argsorted[ :num_rows_to_discard ]
			uncorrelated_robust = numpy.delete( uncorrelated, rows_to_discard, axis = 0 )
			print( "Re-running MVES" )
			solution, weights_robust, iter_num = mves2.MVES( uncorrelated_robust, method = args.method, linear_solver = args.linear_solver, max_iter = args.max_iter )
			weights = numpy.dot( numpy.linalg.inv( solution ), numpy.concatenate( ( uncorrelated.T, numpy.ones((1,uncorrelated.shape[0])) ), axis=0 ) ).T
			volume = abs( np.linalg.det( solution ) )
			print( "robust solution" )
			print( solution )
		
		results.append( ( volume, solution, iter_num, weights, Ts_mapper ) )
	
	## TODO Q: min() or max()? min() is good for outliers. max() is better if we want
	##         to avoid losing parts (larger error when restricted to positive weights).
	volume, solution, iter_num, weights, Ts_mapper = max( results )
	print( "=> Best simplex found with volume:", volume )
	
	## Did we randomly sample less than 100%?
	if len( weights ) < len( all_Ts ):
		assert args.random_percent is not None
		weights = mves2.MVES_solution_weights_for_points( solution, Ts_mapper.project( all_Ts ) )
		assert len( weights ) == len( all_Ts )
	
	running_time = (time.time() - startTime)/60
	print("\nOptimization costs: %.2f minutes" %running_time)
	print( "solution simplex volumn: ", simplex_volumn( solution[:-1] ).round(4) )
	
	save_json = False
	if save_json:
		import json
		norm_min = solution.dot(weights).min(axis=0)
		norm_scale = 1./( solution.dot(weights).max(axis=0) - solution.dot(weights).min(axis=0) ).max()
		def normalizer( pts ):
			return ( pts - norm_min ) * norm_scale
		json.save( open("data.json",'w'), { "float_colors": { solution.dot(weights).tolist() } }
		json_vs = solution.T[:-1]
		assert solution.shape[1] == 4
		json.save( open("data-overlay.json",'w'), { "vs": { json_vs.tolist() }, "faces":, json_vs[ array([[0,3,2],[3,1,2],[3,0,1],[1,0,2]]) ] }
	
	recovered = Ts_mapper.unproject( solution[:-1].T )
	print( 'recovered', recovered.shape )
	print( recovered.round(3) )
	
	print( "Minimum weight:", weights.min() )
	print( "Number of points with negative weights (< -1e-5):", ( weights < -1e-5 ).any(axis=1).sum() )
	print( "Number of points with negative weights (< -0.1):", ( weights < -0.1 ).any(axis=1).sum() )
	print( "Number of points with negative weights (< -0.5):", ( weights < -0.5 ).any(axis=1).sum() )
	print( "Number of points with negative weights (< -1):", ( weights < -1 ).any(axis=1).sum() )
	
	output_path = os.path.join(per_vertex_folder, "result.txt")
	if args.output is not None:
		output_path = args.output
	print( "Saving recovered results to:", output_path )
	format_loader.write_result(output_path, recovered, weights, iter_num, running_time, col_major=True)
	
	def check_recovered( recovered, ground ):
		flags = numpy.zeros( len(Tmat), dtype = bool )
		dists = numpy.zeros( len(Tmat) )
		for i, ht in enumerate( recovered ):
			min_dist = numpy.linalg.norm( ht - ground[0] )
			for j, gt in enumerate( ground ): 
				min_dist = min( min_dist, numpy.linalg.norm( ht - gt ) )
				if numpy.allclose(ht, gt, rtol=1e-1, atol=1e-2):
					flags[i] = True
					ground = numpy.delete(ground, j, 0)
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
			print( "#unmatched: ", numpy.nonzero( ~status ) )
			print( "Unmatched recovery:" )
			print( recovered[ numpy.nonzero( ~status ) ].round(4) )
			print( "Unmatched ground truth: " )
			print( remains.round(4) )
		
	

	

