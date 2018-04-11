"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division

import argparse
import time
import numpy as np
import glob
import includes
import format_loader
from trimesh import TriMesh

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser( description='Generate meshes given per-vertex tranformations, weights, and rest pose' )
	parser.add_argument( 'rest_pose', type=str, help='Rest pose (OBJ).')
	parser.add_argument( 'weights', type=str, help='weights. (DMAT)')
	parser.add_argument( 'transformations', type=str, help='folder of transformations.')
	parser.add_argument( '--output-folder', '-O', type=str, help='folder of transformations.')
	args = parser.parse_args()
	
	rest_mesh = TriMesh.FromOBJ_FileName(args.rest_pose)
	rest_vs = np.array(rest_mesh.vs)
	rest_fs = np.array(rest_mesh.faces)
	
	gt_w = format_loader.load_DMAT(args.weights)

	trans_dir = args.transformations
	
	import os, sys
	output_dir = trans_dir
	if args.output_folder is not None:
		output_dir = aargs.output_folder
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
	
	paths = glob.glob(trans_dir + "/*.DMAT")
#	paths.sort()
#	positions = np.array([ format_loader.load_DMAT(transform_path).T for transform_path in paths ])
	
	for path in paths:
		name = os.path.splitext( os.path.split( path )[1] )[0]
		deformed_vs = []
		transformations = format_loader.load_DMAT(path).T
		for i, transformation in enumerate( transformations ):
			tran = transformation.reshape(4,3).T
			p = rest_vs[i]
			deformed_vs.append( np.dot(tran[:, :3], p[:,np.newaxis]).squeeze() + tran[:, 3] )
			
		output_path = os.path.join( output_dir, name + ".obj" )
		print( "save deformed pose at ", output_path )
		format_loader.write_OBJ( output_path, deformed_vs, rest_fs )
	
	