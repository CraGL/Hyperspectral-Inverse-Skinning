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
import numpy
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
	
	## output name prefix
	output_prefix="per_vertex_recovering"

	## coefficient weights for optimization
	weights = {'W_svd': 2.0, 
		   'W_rotation': 0.01, 
		   'W_rotation1': 0.0, 
		   'W_rotation2': 0.0, 
		   'W_translation':0.0, 
		   'W_spatial': 0.0
		  }
		  	
	mesh_List = []
	if args[1] == '--ssd':
		print("Import poses: ", args[2])
		poses = load_poses( args[2] )
		for i in range( len(poses) ):
			mesh_List.append( poses[i].reshape(-1,3) )
			
		start=time.time()
		res=run_one_ssd(mesh, mesh_List, output_prefix, weights, None)
		res=res.reshape((N,-1))
		end=time.time()
		print("using time: ", end-start)
		
	else:
		base_dir = args[2]
		filenames=glob.glob(base_dir+"/*.obj")
		for f in filenames:
			print("Import pose: ", f)
		for i in range(len(filenames)):
			mesh_List.append(TriMesh.FromOBJ_FileName(filenames[i]))	
			
		start=time.time()
		res=run_one(mesh, mesh_List, output_prefix, weights, None)
		res=res.reshape((N,-1))
		end=time.time()
		print("using time: ", end-start)
	
	per_vertex = res
	print( "per-vertex transformation shape: ", per_vertex )
		
