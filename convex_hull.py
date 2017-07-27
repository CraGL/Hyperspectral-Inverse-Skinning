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

def read_DMAT(path):
	lines = [line.strip() for line in open(path, 'r')]
	num_dims,num_verts = numpy.asarray(lines[0].split(' '), dtype=numpy.int64)
	assert(num_dims == 12)		# transformation per vertex
		
	T = numpy.asarray(lines[1:], dtype=numpy.float64)
	
	assert(len(T) == num_dims*num_verts)
	
	## Careful: DMAT is column-major.
	T = T.reshape(num_dims, num_verts).T

	return T
	
def parse_args(parser=None):
	"""
	Uses ArgumentParser to parse the command line arguments.
	Input:
		parser - a precreated parser (If parser is None, creates a new parser)
	Outputs:
		Returns the arguments as a tuple in the following order:
			(in_mesh, in_texture, out_texture, loadFromDirectory, loadFromData,
			 method, sv_method)
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
	for mesh in in_meshes:	print(mesh)
	meshes = [ obj_reader.quads_to_triangles(obj_reader.load_obj(in_mesh)) for in_mesh in in_meshes ]
	
	in_transformations = glob.glob(path + "/*.DMAT")
	Ts = numpy.array([ read_DMAT(transform_path) for transform_path in in_transformations ])
	
	num_poses = Ts.shape[0]
	num_verts = Ts.shape[1]
	
	Ts = numpy.swapaxes(Ts,0,1)
	Ts = Ts.reshape(num_verts, -1)
	
	return (meshes, Ts)
			

## Formalize the above with functions, from Yotam's experiments
def uncorrellated_space( Ts, threshold = 1e-6 ):
	X = Ts
	
	## Subtract the average.
	Xavg = numpy.average( X, axis = 0 )[numpy.newaxis,:]
	print("Xavg: ", Xavg)
	Xp = X - Xavg
	
	U, s, V = numpy.linalg.svd( Xp, full_matrices = False, compute_uv = True )
	
	## The first index less than threshold
	stop_s = len(s) - numpy.searchsorted( s[::-1], threshold )
	print( "s: ", s )
	print( "stop_s: ", stop_s )
		
	return numpy.dot( Xp, V[:stop_s].T )
				
########################################
# CMD-line tool for getting filenames. #
########################################
if __name__ == '__main__':

	# Time the amount of time this takes.
	startTime = time.time()
	(meshes, Ts) = parse_args()
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	print("\nReading pose meshes costs: %.2f seconds" % (time.time() - startTime))
	readTime = time.time()
	
	print( 'Ts.shape:', Ts.shape )
	Ts_unique, unique_to_original_map = uniquify_points_and_return_input_index_to_unique_index_map( Ts, digits = 5 )
	Ts_unique = numpy.asarray( Ts_unique )
	print( 'Unique Ts.shape:', Ts_unique.shape )
	
	uncorrelated = uncorrellated_space( Ts )
	print( uncorrelated.shape )
	
	import scipy.spatial
	hull = scipy.spatial.ConvexHull( uncorrelated )
	print( len( hull.vertices ) )
	print( hull.vertices )

#	  print("Runtime: %g" % (time.time() - startTime), file)

	print("\nTotal Runtime: %.2f seconds" % (time.time() - startTime))

