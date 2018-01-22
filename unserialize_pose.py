from __future__ import print_function, division
from recordclass import recordclass

import numpy as np
from trimesh import TriMesh
import format_loader

if __name__ == "__main__":
	
	import sys, os
	if len( sys.argv ) != 4:
		print( "Usage:", argv[0], "path/to/SSD_input.obj path/to/SSD_input.txt path/to/output_folder" )
		sys.exit(-1)
	rest_path, input_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]
	if output_folder[-1] != '/':
		output_folder += '/'
		
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	mesh = TriMesh.FromOBJ_FileName(rest_path)
	
	first_name = "mesh_0000"
	with open( input_path ) as f:
		for i, line in enumerate( f ):
			if i == 0:
				dims = map( int, line.strip().split() )
				P, N = dims[0], int(dims[1]/3)
			else:
				vs = np.array(map( float, line.strip().split() )).reshape(N, 3)
				name = first_name[:-len(str(i-1))]+str(i-1)
				format_loader.write_OBJ( output_folder+name+".obj", vs, mesh.faces )
			
	
	
	
	
	