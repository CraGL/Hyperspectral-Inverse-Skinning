"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division

import os
import sys
import argparse
import time
import numpy as np
import glob
import includes
import format_loader

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser( description='Compact ground truth files into one txt file.' )
	parser.add_argument( 'gt_folder', type=str, help='Folder containing ground truth transformations.')
	parser.add_argument( '--output', '-O', type=str, help='path to output the error.')

	args = parser.parse_args()
	
	gt_dir = args.gt_folder

	gt_paths = glob.glob(gt_dir + "/*.DMAT")
	gt_paths.sort()
	gt_data = np.array([ format_loader.load_DMAT(transform_path).T for transform_path in gt_paths ])
	gt_data = np.swapaxes(gt_data,0,1).reshape(gt_data.shape[1],-1)
	
	if args.output is not None:
		np.savetxt(args.output, gt_data)
