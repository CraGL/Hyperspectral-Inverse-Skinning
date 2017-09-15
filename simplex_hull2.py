"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""
from __future__ import print_function, division
from recordclass import recordclass

import os
import sys
import time
import numpy

########################################
# CMD-line tool for getting filenames. #
########################################

if __name__ == '__main__':

	# Time the amount of time this takes.
	import simplex_hull
	startTime = time.time()
	(meshes, Ts, Tmat) = simplex_hull.parse_args()
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	print("\nReading pose meshes costs: %.2f seconds" % (time.time() - startTime))
	readTime = time.time()
	
# 	print( 'Ts.shape:', Ts.shape )
# 	Ts_unique, unique_to_original_map = uniquify_points_and_return_input_index_to_unique_index_map( Ts, digits = 5 )
# 	Ts_unique = numpy.asarray( Ts_unique )
# 	print( 'Unique Ts.shape:', Ts_unique.shape )
	
	project, unproject, scale = simplex_hull.uncorrellated_space( Ts )
	uncorrelated = project( Ts )
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
	
	
	## Option 1: HyperCSI
	import HyperCSI
	N = uncorrelated.shape[1] + 1
	solution = HyperCSI.hyperCSI( uncorrelated.T, N )[0]
	
	print( "ground truth simplex volumn:" )
	print( simplex_hull.simplex_volumn( project( Tmat ).T ) )
	print( "solution simplex volumn:" )
	print( simplex_hull.simplex_volumn( solution ) )
 
	## Option 2: compute minimum-volume enclosing simplex
# 	import mves2
# 	solution = mves2.MVES( uncorrelated )
# 	print( "solution" )
# 	print( solution )
	
	recovered = unproject( solution.T )
# 	print( 'recovered' )
# 	print( recovered.round(3) )
	
	def check_recovered( recovered, ground ):
		flags = numpy.zeros( len(Tmat), dtype = bool )
		for i, ht in enumerate( recovered ):
			for j, gt in enumerate( ground ):
				if numpy.allclose(ht, gt, atol=1e-4):
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

