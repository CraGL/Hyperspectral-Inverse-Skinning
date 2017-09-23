"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division
from recordclass import recordclass

import os
import sys
import argparse
import time
import numpy
import scipy

import obj_reader
from trimesh import TriMesh
from scipy.spatial import ConvexHull
import glob

def create_parser():
	""" Creates an ArgumentParser for this command line tool. """
	parser = argparse.ArgumentParser(description = "Minimize the difference " +
		"in sampled values along texture edge pairs.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		usage="%(prog)s path/to/input_model_folder")
	parser.add_argument("pose_folder", metavar="path/to/input_models_folder",
		help="Path to the folder containing input mesh files.")
	
	return parser

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
	
def parse_args(parser=None):
	"""
	Uses ArgumentParser to parse the command line arguments.
	Input:
		parser - a precreated parser (If parser is None, creates a new parser)
	Outputs:
		Returns the arguments as a tuple in the following order:
			(in_mesh, Ts, Tmat)
	"""
	if(parser is None):
		parser = create_parser()
	args = parser.parse_args()

	# Check that in_mesh exists
	if(not os.path.exists(args.pose_folder)):
		parser.error("Path to pose folder does not exist.")
	
	path = args.pose_folder	 
#	  if( path[-1] != '/' ):	path.append('/')
	in_meshes = glob.glob(path + "/*.obj")
	for in_mesh in in_meshes:	print(in_mesh)
#	meshes = [ obj_reader.quads_to_triangles(obj_reader.load_obj(in_mesh)) for in_mesh in in_meshes ]
	meshes = [ TriMesh.FromOBJ_FileName(in_mesh) for in_mesh in in_meshes ]
	
	in_transformations = glob.glob(path + "/*.DMAT")
	import DMAT2MATLAB
	Ts = numpy.array([ DMAT2MATLAB.load_DMAT(transform_path).T for transform_path in in_transformations ])
	
	handle_trans = glob.glob(path + "/*.Tmat")
	Tmat = numpy.array([ DMAT2MATLAB.load_Tmat(transform_path).T for transform_path in handle_trans ])
	
	num_poses = Ts.shape[0]
	num_verts = Ts.shape[1]
	
	Ts = numpy.swapaxes(Ts,0,1)
	Ts = Ts.reshape(num_verts, -1)
	
	Tmat = numpy.swapaxes(Tmat,0,1)
	Tmat = Tmat.reshape(Tmat.shape[0], Tmat.shape[1]*Tmat.shape[2])
	
	return (meshes, Ts, Tmat)

class SpaceMapper( object ):
	def __init__( self ):
		self.Xavg_ = None
		self.U_ = None
		self.s_ = None
		self.V_ = None
		self.stop_s = None
		self.scale = None
	
	def project( self, correllated_poses ):
		scale = self.scale
		stop_s = self.stop_s
		V = self.V_
		Xavg = self.Xavg_
		if scale is not None:
			return numpy.multiply( numpy.dot( correllated_poses - Xavg, V[:stop_s].T ), scale )
		else:
			return numpy.dot( correllated_poses - Xavg, V[:stop_s].T )

	def unproject( self, uncorrellated_poses ):
		scale = self.scale
		stop_s = self.stop_s
		V = self.V_
		Xavg = self.Xavg_
		if scale is not None:
			return numpy.dot( numpy.divide( uncorrellated_poses, scale ), V[:stop_s] ) + Xavg
		else:
			return numpy.dot( uncorrellated_poses, V[:stop_s] ) + Xavg 
	
	def PCA_Dimension( X, threshold = 1e-6 ):
		## Subtract the average.
		X = numpy.array( X )
		Xavg = numpy.average( X, axis = 0 )[numpy.newaxis,:]
		Xp = X - Xavg
	
		U, s, V = numpy.linalg.svd( Xp, full_matrices = False, compute_uv = True )
	
		## The first index less than threshold
		stop_s = len(s) - numpy.searchsorted( s[::-1], threshold )
		
		return stop_s
	
	PCA_Dimension = staticmethod( PCA_Dimension )
	
	## Formalize the above with functions, from Yotam's experiments
	def Uncorrellated_Space( X, enable_scale=True, threshold = 1e-6 ):
		space_mapper = SpaceMapper()
	
		## Subtract the average.
		Xavg = numpy.average( X, axis = 0 )[numpy.newaxis,:]
		# print("Xavg: ", Xavg)
		Xp = X - Xavg
		space_mapper.Xavg_ = Xavg
	
		U, s, V = numpy.linalg.svd( Xp, full_matrices = False, compute_uv = True )
		space_mapper.U_ = U
		space_mapper.s_ = s
		space_mapper.V_ = V
	
		## The first index less than threshold
		stop_s = len(s) - numpy.searchsorted( s[::-1], threshold )
		# print( "s: ", s )
		# print( "stop_s: ", stop_s )
		space_mapper.stop_s = stop_s
	
		## Change scale to something that makes the projection of the points
		## have unit size in each dimension...
		if enable_scale:
			scale = numpy.array( [1./(max(x)-min(x)) for x in numpy.dot( Xp, V[:stop_s].T ).T ] )
			# print( "scale: ", scale )
			space_mapper.scale = scale 
	
		return space_mapper
		
	Uncorrellated_Space = staticmethod( Uncorrellated_Space )

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
		if( len( current_vertex_set + expansion_set ) > Ts.shape[1] ):
			current_points = numpy.take( Ts, current_vertex_set + expansion_set, axis=0 )
			hull = ConvexHull( current_points )
			hull_vertices = numpy.take( hull.points, hull.vertices, axis=0 )
			small_sets.append( hull_vertices )
		
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

	# Time the amount of time this takes.
	startTime = time.time()
	(meshes, Ts, Tmat) = parse_args()
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	print("\nReading pose meshes costs: %.2f seconds" % (time.time() - startTime))
	startTime = time.time()
	
#	print( 'Ts.shape:', Ts.shape )
#	Ts_unique, unique_to_original_map = uniquify_points_and_return_input_index_to_unique_index_map( Ts, digits = 5 )
#	Ts_unique = numpy.asarray( Ts_unique )
#	print( 'Unique Ts.shape:', Ts_unique.shape )
	
	Ts_mapper = SpaceMapper.Uncorrellated_Space( Ts )
	uncorrelated = Ts_mapper.project( Ts )
	print( "uncorrelated data shape" )
	print( uncorrelated.shape )
	Ts_sets = divide_mesh_into_small_sets(meshes[0], uncorrelated)
#	N = len( uncorrelated )
#	Ts_sets = [ uncorrelated[:N/4], uncorrelated[N/4:N/2], uncorrelated[N/2:N*3/4], uncorrelated[N*3/4:] ]
# 	print( "# small sets: ", len( Ts_sets ) )
	print("\nDividing mesh into small sets costs: %.2f seconds" % (time.time() - startTime))
	startTime = time.time()
	
#	import os,sys
#	dirs = sys.argv[-1].split(os.path.sep)
#	assert( len(dirs) > 2 )
#	path = "benchmarks/" + dirs[-2] + "-" + dirs[-1] + ".csv"
#	print("path: ", path)
#	numpy.savetxt(path, uncorrelated, fmt="%1.6f", delimiter=",")
#	outpath = "benchmarks/" + dirs[-2] + "-" + dirs[-1] + ".mat"
#	import scipy.io
#	scipy.io.savemat( outpath, { 'X': uncorrelated } )
	
	numpy.set_printoptions(precision=4, suppress=True)
	print( "ground truth simplex volumn:" )
	print( simplex_volumn( Ts_mapper.project( Tmat ).T ) )
 
	## plot uncorrelated data
#	import matplotlib.pyplot as plt
#	from mpl_toolkits.mplot3d import Axes3D
#	
#	plt3d = plt.figure().gca(projection='3d')
#	plt3d.scatter(uncorrelated[:,0] , uncorrelated[:,1] , uncorrelated[:,2],  color='green')
#	plt.show()
 
	## Compute minimum-volume enclosing simplex
	import mves2
# 	solution = mves2.MVES( uncorrelated )
# 	print( "solution 1: ", solution )

	## Option 1: Run MVES on the union of convex hull vertices:
	good_verts = []
	for verts in Ts_sets:
		mapper = SpaceMapper.Uncorrellated_Space( verts, False )
		hull = ConvexHull( mapper.project( verts ) )
		projected_verts = numpy.take( hull.points, hull.vertices, axis=0 )
		if good_verts == []: 
			good_verts = mapper.unproject( projected_verts )
		else:
			good_verts = numpy.concatenate( (good_verts, mapper.unproject( projected_verts )), axis=0 )
	solution = mves2.MVES( good_verts )
	
	## Option 2: Run MVES on the small sets
# 	good_verts = []
# 	for verts in Ts_sets:
# 		if good_verts == []:
# 			good_verts = mves2.MVES( verts ).x[:-1].T
# 		else:
# 			good_verts = numpy.concatenate( (good_verts, mves2.MVES( verts ).x[:-1].T), axis=0 )
# 	solution = mves2.MVES( good_verts )
	
	print( "solution" )
	print( solution )
	print("\nOptimization costs: %.2f seconds" % (time.time() - startTime))
	print( "solution simplex volumn: ", simplex_volumn( solution.x[:-1] ) )
	
	recovered = Ts_mapper.unproject( solution.x[:-1].T )
#	print( 'recovered' )
#	print( recovered.round(3) )
	
	def check_recovered( recovered, ground ):
		flags = numpy.zeros( len(Tmat), dtype = bool )
		for i, ht in enumerate( recovered ):
			for j, gt in enumerate( ground ):
				if numpy.allclose(ht, gt, atol=1e-3):
					flags[i] = True
					ground = numpy.delete(ground, j, 0)
					break
		return flags, ground
	
	status, remains = check_recovered( recovered, Tmat )
	if( all( status ) ):	print( "match ground truth" )
	else:
		print( "#unmatched: ", numpy.nonzero( ~status ) )
		print( "Unmatched recovery:" )
		print( recovered[ numpy.nonzero( ~status ) ].round(4) )
		print( "Unmatched ground truth: " )
		print( remains.round(4) )

	

