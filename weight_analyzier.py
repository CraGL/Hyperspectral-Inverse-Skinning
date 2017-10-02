"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division
from recordclass import recordclass

import os
import sys
import numpy
import DMAT2MATLAB

if __name__ == '__main__':
	if len( sys.argv ) != 2:
		print( 'Usage:', sys.argv[0], 'path/to/input.DMAT', file = sys.stderr )
		sys.exit(-1)
		
	path = sys.argv[1]
	W = DMAT2MATLAB.load_DMAT( path )
	
	print( "Load weight from ", path )
	print( "Weights have a shape of ", W.shape )
	
	print( "Max weight per handle: " )
	print( numpy.amax( W, axis = 1 ) )
	
	print( "Min weight per handle: " )
	print( numpy.amin( W, axis = 1 ) )
	
	print( "Sum weight per vertex: " )
	print( numpy.sum( W, axis = 0 ) )
	
	import simplex_hull
	NN = simplex_hull.SpaceMapper.PCA_Dimension( W )
	print( "PCA dimensions: ", NN )
	
	print( "Purity rho: " )
	print( numpy.amax( [ numpy.linalg.norm( row ) for row in W.T ] ) )