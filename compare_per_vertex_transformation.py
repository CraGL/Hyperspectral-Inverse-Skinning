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
import glob
import includes
import format_loader

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser( description='Compare our recovered transformations with ground truth.' )
	parser.add_argument( 'gt_folder', type=str, help='Folder containing ground truth transformations.')
	parser.add_argument( 'rev_folder', type=str, help='Folder containing recovered transformations.')
	parser.add_argument( '--output', type=str, help='path to output the error.')
	def str2bool(s): return {'true': True, 'false': False}[s.lower()]
	parser.add_argument( '--write-OBJ', '-W', type=str2bool, default=False, help='whether to save recovered poses.')

	args = parser.parse_args()
	
	gt_dir = args.gt_folder
	rev_dir = args.rev_folder

	gt_paths = glob.glob(gt_dir + "/*.DMAT")
	gt_paths.sort()
	gt_data = np.array([ format_loader.load_DMAT(transform_path).T for transform_path in gt_paths ])
	
	rev_paths = glob.glob(rev_dir + "/*.DMAT")
	rev_paths.sort()
	rev_data = np.array([ format_loader.load_DMAT(transform_path).T for transform_path in rev_paths ])

	
	def compute_error( gt, data ):
		error = []
		for pose_gt, pose_data in zip(gt, data):
			error.append( np.array([np.linalg.norm(pt_gt - pt_data) for pt_gt, pt_data in zip(pose_gt, pose_data)]) )
		
		## Divide this by the bounding sphere radius (or 1/2 the bounding box diagonal?)
		## to get the error metric E_RMS from Kavan 2010.
		E_RMS_kavan2010 = 1000*np.linalg.norm( gt.ravel() - data.ravel() )/np.sqrt(3*gt.shape[0]*gt.shape[1])
		## We are assuming that `gt` is poses-by-#vertices-by-3.
		assert len( gt.shape ) == 3
		assert gt.shape[2] == 3
		
		return np.array(error), E_RMS_kavan2010
	
	
	print( "############################################" )
	print( "Per-vertex transformation Error: " )
	error = np.array([[np.linalg.norm(gt_pt - rev_pt) for gt_pt, rev_pt in zip(gt_pose, rev_pose)] for gt_pose, rev_pose in zip(gt_data, rev_data)])
	print( "rev error: ", np.linalg.norm(error)/error.size )	
	print( "max, mean and median per-vertex transformation error", np.max(error), np.mean(error), np.median(error) )
# 	print( "SSD E_RMS_kavan2010: ", ssd_erms )
	print( "############################################" )
	
	import os, sys
	output_dir = None
	if args.output is not None:
		output_dir = args.output
		output_path = os.path.join( output_dir, "compare_per_vertex_tranformation.out" )
		format_loader.write_DMAT( output_path, error )
		

 
	
	
	
		
