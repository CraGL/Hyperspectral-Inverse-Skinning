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
import format_loader

from trimesh import TriMesh
	
def lbs_all(rest, bones, W):
	'''
	bones are a list of flattened row-major matrices
	'''
	assert( len(bones.shape) == 2 )
	assert( len(W.shape) == 2 )
	assert( bones.shape[0] == W.shape[1] )
	assert( bones.shape[1] == 12 )
	assert( W.shape[0] == rest.shape[0] )
	assert( rest.shape[1] == 3 )
	
	per_vertex = np.dot(W, bones)
	N = rest.shape[0]
	result = np.zeros((N,3))
	
	for i in range(N):
		tran = per_vertex[i].reshape(3,4)
		p = rest[i]
		result[i] = np.dot(tran[:, :3], p[:,np.newaxis]).squeeze() + tran[:, 3]
		
	return result
	

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser( description='Compare our results and SSD approch results with ground truth.' )
	parser.add_argument( 'rest_pose', type=str, help='Rest pose (OBJ).')
	parser.add_argument( 'result', type=str, help='our results(txt).')
	parser.add_argument( '--output', '-O', type=str, help='path to save recovered poses.')
	args = parser.parse_args()
		
	rest_mesh = TriMesh.FromOBJ_FileName(args.rest_pose)
	rest_vs = np.array(rest_mesh.vs)
	rest_fs = np.array(rest_mesh.faces)
		
	rev_bones, rev_w = format_loader.load_result(args.result)
	np.set_printoptions(precision=6, suppress=True)
				
	## Adjust bones data to Pose-by-bone-by-transformation
	rev_bones = np.swapaxes(rev_bones, 0, 1)
	rev_vs = np.array([lbs_all(rest_vs, rev_bones_per_pose, rev_w.T) for rev_bones_per_pose in rev_bones ])	
	
	output_folder = os.path.split(args.result)[0] + "/our_recovered"		
	if args.output is not None:
		output_folder = args.output
	
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
		
	for i, vs in enumerate(rev_vs):
		output_path = os.path.join(output_folder, str(i+1) + ".obj")
		format_loader.write_OBJ( output_path, vs.round(6), rest_fs )

 
	
	
	
		
