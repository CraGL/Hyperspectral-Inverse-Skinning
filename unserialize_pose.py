from __future__ import print_function, division
from recordclass import recordclass

import numpy as np
import trimesh

if __name__ == "__main__":
	
	import sys, os
	if len( sys.argv ) != 4:
		print( "Usage:", argv[0], "path/to/SSD_input.obj path/to/SSD_input.txt path/to/output_folder" )
		sys.exit(-1)
	rest_path, input_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]

	mesh = trimesh.FromOBJ_FileName(rest_path)
	
	P=0, N=0
	with open( input_path ) as f:
		for i, line in enumerate( f ):
			if i == 0:
				dims = map( int, line.strip().split() )
				P, N = dims[0], dims[1]/3
			else:
				vs = map( float, line.strip().split() ).reshape(N, 3)
				mesh.vs = list( vs )
				
				mesh.write
			
	
# 	np.savetxt( output_file, data_set, fmt='%.6f', delimiter=' ', header=shape, comments='')
	
	
	
	