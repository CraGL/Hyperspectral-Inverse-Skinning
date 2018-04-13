"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division
# from recordclass import recordclass

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

	
	def transformation_matrix_error(gt, data):
	    diff=abs(gt-data).ravel()
	    rmse=np.sqrt(np.square(gt-data).sum()/len(gt.ravel()))
	    return [max(diff), min(diff), np.median(diff), np.mean(diff), rmse]
	    
	
	
	print( "############################################" )
	print( "Per-vertex transformation Error: " )
	print( "max, min, median, mean and rmse per-vertex transformation error:")
	errors=transformation_matrix_error(gt_data, rev_data)
	print (errors)
	print( "############################################" )


	
	import os, sys
	output_dir = None
	if args.output is not None:
		output_dir = args.output
		output_path = os.path.join( output_dir, "compare_per_vertex_tranformation.out" )
		format_loader.write_DMAT( output_path, error )
		

 
	
	
	
		
