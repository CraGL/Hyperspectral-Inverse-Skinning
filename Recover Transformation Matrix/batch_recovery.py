from __future__ import print_function, division
from recordclass import recordclass

import numpy as np
import json
import time
import scipy
import scipy.sparse
from trimesh import TriMesh
import glob
import single_recovrey
	
if __name__ == '__main__':
	np.set_printoptions(linewidth=2000, suppress=True)
	from Extract_Transformation_matrix_minimize_SVD_singular_value import *

	#### read obj file
	import sys
	if len( sys.argv ) != 3:
		print( "usage: ", sys.argv[0], "path/to/rest_pose, path/to/poses" )
		exit(-1)
	
	## fetch the poses and ground truth if given
	rest_pose_path, pose_dir = sys.argv[1], sys.argv[2]
	groundtruth_paths = glob.glob(pose_dir+"/*.DMAT")
	pose_paths = glob.glob(pose_dir+"/*.obj")
	print( "Load filenames: ", pose_paths )
	print( "Load weight ground truth: ", groundtruth_paths )
	P = len(pose_paths)
	assert( len(groundtruth_paths) == 0 or len(groundtruth_paths) == P )
	rest_pose = TriMesh.FromOBJ_FileName(rest_pose_path)
	
	meshes_List, groundtruth_List = [], []
	for path in pose_paths:
		meshes_List.append(TriMesh.FromOBJ_FileName(path))
	for path in groundtruth_paths:
		groundtruth_List.append(load_DMAT(sys.argv[3]).T)

	#### solve for transformation per vertex, with smoothness regularization and transformation matrix L2 norm regularization.
	for i in range(P):
		single_recovrey.compare_one_pose(rest_pose, meshes_List[i], groundtruth_paths[i])

	