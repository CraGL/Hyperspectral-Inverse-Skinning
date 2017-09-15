#!/usr/bin/python

# Convert a libigl DMAT to a Matlab .mat file.
# Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
# License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)
# On GitHub as a gist: https://gist.github.com/yig/0fb7fe73b2ce914c4b1d6de3b4e4ba01

from __future__ import print_function, division

from numpy import *
import scipy.io
import sys

def load_DMAT( path ):
    from numpy import zeros
    
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
    from numpy import zeros
    
    with open( path ) as f:
        v = []
        for i, line in enumerate( f ):
            v = v + list( map( float, line.strip().split() ) )
         
        M = array(v)
    
    M = M.reshape( -1, 12 ).T
    
    return M

if __name__ == '__main__':
	if len( sys.argv ) != 3:
		print( 'Usage:', sys.argv[0], 'path/to/input.DMAT path/to/output.mat [matlab_variable_name]', file = sys.stderr )
		sys.exit(-1)

	argv = list( sys.argv[1:] )
	inpath = argv.pop(0)
	outpath = argv.pop(0)

	matlab_variable_name = 'X'
	if len( argv ) > 0:
		matlab_variable_name = argv.pop(0)

	assert len( argv ) == 0

	print( "Loading:", inpath )
	X = load_DMAT( inpath )
	print( "Loaded:", inpath )
	print( "Saving:", outpath )
	scipy.io.savemat( outpath, { matlab_variable_name: X } )
	print( "Saved:", outpath )
