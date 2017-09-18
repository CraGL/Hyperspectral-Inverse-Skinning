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
	
	## Change scale to something that makes the projection of the points
	## have unit size in each dimension...
	# scale = array( [numpy.linalg.norm(x) for x in project( Ts ).T ] )
	scale = numpy.array( [1./(max(x)-min(x)) for x in numpy.dot( Xp, V[:stop_s].T ).T ] )
	print( "scale: ", scale )
# 	scale = max( scale )
	
	def project( correllated_poses, scale = None ):
		if scale is not None:
			return numpy.multiply( numpy.dot( correllated_poses - Xavg, V[:stop_s].T ), scale )
		else:
			return numpy.dot( correllated_poses - Xavg, V[:stop_s].T )
	
	def restore( uncorrellated_poses, scale = None ):
		if scale is not None:
			return numpy.dot( numpy.divide( uncorrellated_poses, scale ), V[:stop_s] ) + Xavg
		else:
			return numpy.dot( uncorrellated_poses, V[:stop_s] ) + Xavg
	
	return project, restore, scale

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


########################################
# CMD-line tool for getting filenames. #
########################################
if __name__ == '__main__':

	# Time the amount of time this takes.
	startTime = time.time()
	(meshes, Ts, Tmat) = parse_args()
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	print("\nReading pose meshes costs: %.2f seconds" % (time.time() - startTime))
	readTime = time.time()
	
# 	print( 'Ts.shape:', Ts.shape )
# 	Ts_unique, unique_to_original_map = uniquify_points_and_return_input_index_to_unique_index_map( Ts, digits = 5 )
# 	Ts_unique = numpy.asarray( Ts_unique )
# 	print( 'Unique Ts.shape:', Ts_unique.shape )
	
	project, unproject, scale = uncorrellated_space( Ts )
	uncorrelated = project( Ts, scale )
	print( "uncorrelated data shape" )
	print( uncorrelated.shape )
	
	import os,sys
	dirs = sys.argv[-1].split(os.path.sep)
	assert( len(dirs) > 2 )
	path = "benchmarks/" + dirs[-2] + "-" + dirs[-1] + ".csv"
	print("path: ", path)
	numpy.savetxt(path, uncorrelated, fmt="%1.6f", delimiter=",")
	outpath = "benchmarks/" + dirs[-2] + "-" + dirs[-1] + ".mat"
	import scipy.io
	scipy.io.savemat( outpath, { 'X': uncorrelated } )
	
	numpy.set_printoptions(precision=4, suppress=True)
	print( "ground truth simplex volumn:" )
	print( simplex_volumn( project( Tmat, scale ).T ) )
	
	## Option 1: HyperCSI
# 	import HyperCSI
# 	N = uncorrelated.shape[1] + 1
# 	solution_1 = HyperCSI.hyperCSI( uncorrelated.T, N )[0]
 
	## Option 2: compute minimum-volume enclosing simplex
	import mves2
	solution = mves2.MVES( uncorrelated )
# 	solution = mves2.MVES( uncorrelated, project( Tmat ) )
	print( "solution simplex volumn:" )
	print( simplex_volumn( solution.x[:-1] ) )
	print( "solution" )
	print( solution )
	
	recovered = unproject( solution.x[:-1].T, scale )
# 	print( 'recovered' )
# 	print( recovered.round(3) )
	
	def check_recovered( recovered, ground ):
		flags = numpy.zeros( len(Tmat), dtype = bool )
		for i, ht in enumerate( recovered ):
			for j, gt in enumerate( ground ):
				if numpy.allclose(ht, gt, atol=1e-3):
					flags[i] = True
					numpy.delete(ground, j, 0)
					break
		return flags, ground
	
	status, remains = check_recovered( recovered, Tmat )
	if( all( status ) ):	print( "match ground truth" )
	else:
		print( "#unmatched: ", numpy.nonzero( ~status ) )
		print( "Unmatched recovery:" )
		print( recovered[ numpy.nonzero( ~status ) ].round(3) )
		print( "Unmatched ground truth: " )
		print( remains.round(3) )
#	  print("Runtime: %g" % (time.time() - startTime), file)

	print("\nTotal Runtime: %.2f seconds" % (time.time() - startTime))

