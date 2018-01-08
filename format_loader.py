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
	
def write_DMAT( path, M ):
	assert( len( M.shape ) == 2 and "M is a matrix")
	rows, cols = M.shape
	with open( path, 'w' ) as f:
		f.write(repr(cols) + " " + repr(rows) + "\n")
		for i in range(cols):
			for e in M[:,i]:
				f.write(repr(e) + "\n")
	
def load_Tmat( path ):	
	with open( path ) as f:
		v = []
		for i, line in enumerate( f ):
			v = v + list( map( float, line.strip().split() ) )
		 
		M = array(v)
	
	M = M.reshape( -1, 12 )
	
	return M
	
def load_poses( path ):	
	'''
	Load the SSD input txt file.
	'''
	M = []
	dims = None
	with open( path ) as f:
		for i, line in enumerate( f ):
			if i == 0:
				dims = list( map( int, line.strip().split() ) )
			
			else:
				M.append( list( map( float, line.strip().split() ) ) )
	
	M = asarray( M )
	M = M.reshape( (dims[0], -1, 3) )
	
	return M
	
def load_result( path ):
	'''
	Load the SSD output txt file.
	'''
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

def write_result(path, res, weights, iter_num, time, col_major=False):
	'''
	write recovered per-bone tranformation matrix and weights following the SSD output format.
	The bone transformations are flattened row-major matrices.
	'''
	B = len(res)
	res = res.reshape(B,-1,12)
	if col_major:
		res = res.reshape(B,-1,4,3)
		res = swapaxes( res,2,3 )
		res = res.reshape(B,-1,12)
		
	nframes = len(res[0])
	with open( path, 'w' ) as f:
		f.write("#####################################################\n")
		f.write("# (C) Songrun (songruner@gmail.com)\n")
		f.write("#\n")
		f.write("# Running time: " + repr(round(time, 3)) + " (s)\n")
		f.write("# Repeat      : " + repr(iter_num) + "\n")
		f.write("#\n")
		f.write("#####################################################\n")
		for i in range(B):
			f.write("*BONEANIMATION, BONEINDEX=" + repr(i) + ", NFRAMES=" + repr(nframes) + "\n")
			for j in range(nframes):
				s = repr(j)
				for k in range( 12 ):
					s = s + " " + repr(res[i,j,k])
				s += " 0 0 0 1\n"
				f.write(s)
			f.write("#####################################################\n")
			
		## write weights
		m, n = weights.shape[0], weights.shape[1]
		f.write("*VERTEXWEIGHTS, NVERTICES=" + repr(m) + "  #(vtx0Based bone0 w0 bone1 w1 ... )\n")	
		for i in range(m):
			s = repr(i)+" "
			for j in range(n):
				s = s + " " + repr(j) + " " + repr(weights[i,j])
			s += "\n"
			f.write(s)
		f.write("#####################################################\n")
		

def write_OBJ( path, vs, fs ):
	with open( path, 'w' ) as file:
		for v in vs:
			file.write("v " + repr(v[0]) + " " + repr(v[1]) + " " + repr(v[2]) + "\n")	
		for f in fs:
			file.write("f " + repr(f[0]+1) + " " + repr(f[1]+1) + " " + repr(f[2]+1) + "\n")


## test load poses
if __name__ == '__main__':
	if len( sys.argv ) != 2:
		print( 'Usage:', sys.argv[0], 'path/to/poses.txt', file = sys.stderr )
		sys.exit(-1)

	M = load_poses(sys.argv[1])
	print( "poses", M.shape )
	print( M )
