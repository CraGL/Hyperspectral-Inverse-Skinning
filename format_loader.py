#!/usr/bin/python

# Convert a libigl DMAT to a Matlab .mat file.
# Author: Yotam Gingold <yotam (strudel) yotamgingold.com>, Songrun Liu<songruner @ gmail.com>
# License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)
# On GitHub as a gist: https://gist.github.com/yig/0fb7fe73b2ce914c4b1d6de3b4e4ba01

from __future__ import print_function, division

from numpy import *
import scipy.io
import sys

def load_DMAT( path ):
	with open( path ) as f:
		
		for i, line in enumerate( f ):
			if i == 0:
				dims = list( map( int, line.strip().split() ) )
				M = zeros( prod( dims ) )
			
			else:
				M[i-1] = float( line )
	
	M = M.reshape( dims )
	
	return M
	
def load_Tmat( path ):	
	with open( path ) as f:
		v = []
		for i, line in enumerate( f ):
			v = v + list( map( float, line.strip().split() ) )
		 
		M = array(v)
	
	M = M.reshape( -1, 12 ).T
	
	return M
	
def load_poses( path ):	
	M = []
	with open( path ) as f:
		for i, line in enumerate( f ):
			if i == 0:
				dims = list( map( int, line.strip().split() ) )
			
			else:
				M.append( list( map( float, line.strip().split() ) ) )
	
	M = asarray( M )
	return M
	
def load_result( path ):
	## M is Bone-by-Frame-by-12
	M = []
	W = None
	section = None
	count, B, nframes, rows = 0, 0, 0, 0
	with open( path ) as f:
		for line in f:
			words = line.strip().split(", ")
					
			if len(words) == 3 and words[0] == "*BONEANIMATION":
				section = "bone"
				nframes = int(words[2][len("NFRAMES="):])
				count=0
				B += 1
				
			elif len(words) > 0 and words[0] == "*VERTEXWEIGHTS":
				section = "weight"
				rows = int(words[1].split(" ")[0].split("=")[1])
				W = zeros((rows, B))
				count = 0
			
			elif section == "bone" and count < nframes:
				words = line.strip().split(" ")
				assert( len(words) == 17 )
				M.append( list( map(float, words[1:13] ) ) )
				count+=1
			
			elif section == "weight" and count < rows:
				words = line.strip().split(" ")
				assert( len( words ) % 2 == 0 )
				ridx = int( words[0] )
				for i in range( int(len(words[2:])/2) ):
					cidx = int( words[i*2+2] )
					val = float( words[i*2+3] )
					W[ridx, cidx] = val
				count+=1
	
	M = asarray( M ).reshape(B,-1,12)
	
	return M, W.T


## test load DMAT
# if __name__ == '__main__':
#	if len( sys.argv ) != 3:
#		print( 'Usage:', sys.argv[0], 'path/to/input.DMAT path/to/output.mat [matlab_variable_name]', file = sys.stderr )
#		sys.exit(-1)
# 
#	argv = list( sys.argv[1:] )
#	inpath = argv.pop(0)
#	outpath = argv.pop(0)
# 
#	matlab_variable_name = 'X'
#	if len( argv ) > 0:
#		matlab_variable_name = argv.pop(0)
# 
#	assert len( argv ) == 0
# 
#	print( "Loading:", inpath )
#	X = load_DMAT( inpath )
#	print( "Loaded:", inpath )
#	print( "Saving:", outpath )
#	scipy.io.savemat( outpath, { matlab_variable_name: X } )
#	print( "Saved:", outpath )

## test load poses
if __name__ == '__main__':
	if len( sys.argv ) != 2:
		print( 'Usage:', sys.argv[0], 'path/to/poses.txt', file = sys.stderr )
		sys.exit(-1)

	M = load_poses(sys.argv[1])
	print( "poses", M.shape )
	print( M )
