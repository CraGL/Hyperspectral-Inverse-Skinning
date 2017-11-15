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
import numpy as np
import scipy
import glob
import includes

from trimesh import TriMesh
from format_loader import load_poses
from space_mapper import SpaceMapper
from PerVertex.Extract_Transformation_matrix_minimize_SVD_singular_value_allPoses import *

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print( 'Usage:', sys.argv[0], 'path/to/restpose.obj', 'path/to/poses.txt', file = sys.stderr )
		exit(-1)
	args = sys.argv[1:]
	assert( args[1] == '--ssd' or args[1] == '--obj' )
	
	## mesh at rest pose
	print("Import mesh: ", args[0])
	mesh = TriMesh.FromOBJ_FileName(args[0])
	N = len(mesh.vs)	
		  	
	all_poses = []
	if args[1] == '--ssd':
		print("Import poses: ", args[2])
		poses = load_poses( args[2] )
		for i in range( len(poses) ):
			all_poses.append( poses[i].reshape(-1,3) )
					
	else:
		base_dir = args[2]
		filenames=glob.glob(base_dir+"/*.obj")
		for f in filenames:
			print("Import pose: ", f)
		mesh_list = [TriMesh.FromOBJ_FileName(filenames[i]) for in range(len(filenames)]
		all_poses = [ mesh.vs for mesh in mesh_list ]
	
	all_poses = np.asarray( all_poses )
	
	
		
