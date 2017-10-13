from __future__ import print_function, division
from recordclass import recordclass

import numpy as np
import glob
import obj_reader

if __name__ == "__main__":
	
	import sys, os
	if len( sys.argv ) != 3:
		print( "Usage:", argv[0], "path/to/pose_dir, output_name" )
		sys.exit(-1)
	dir_path, output = sys.argv[1], sys.argv[2]
	
	mesh_paths = glob.glob(dir_path + "/*.obj")
	
	data_set = []
	for path in mesh_paths:	
		print(path)
		vs = obj_reader.load_obj(path).v
		vs = np.array( vs ).flatten()
		data_set.append( vs )
		
	data_set = np.array( data_set )
	
	nRow, nCol = data_set.shape[0], data_set.shape[1]
	output_file = dir_path + '/' + output + '.txt'
	with open(output_file, 'w') as f:
		f.write(str(nRow) + ' ' + str(nCol) + '\n')
		for i in range(nRow):
			f.write("%.10f"%data_set[i,0])
			for j in range(1,nCol):
				f.write(' ' + "%.10f"%data_set[i,j])
			f.write('\n')
		
	
# 	np.savetxt( output_file, data_set, fmt='%.6f', delimiter=' ', header=shape, comments='')
	
	
	
	