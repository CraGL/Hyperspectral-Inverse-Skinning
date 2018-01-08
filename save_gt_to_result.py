from __future__ import print_function, division

import os
import sys
import argparse
import glob
import format_loader
import numpy

if __name__ == '__main__':

	'''
	Uses ArgumentParser to parse the command line arguments.
	Input:
		parser - a precreated parser (If parser is None, creates a new parser)
	Outputs:
		Returns the arguments as a tuple in the following order:
			(in_mesh, Ts, Tmat)
	'''
		
	parser = argparse.ArgumentParser(description = "Save ground truth to result.txt. ")
	parser.add_argument("per_bone", type=str, help="Path to the folder containing ground truth per-bone transformation.")
	parser.add_argument('weights', type=str, help='Ground truth weights path.')
	parser.add_argument('output', type=str, help='output path.')
	args = parser.parse_args()
	
	per_vertex_folder = args.per_bone
	handle_trans = glob.glob(per_vertex_folder + "/*.Tmat")
	handle_trans.sort()
	Tmat = numpy.array([ format_loader.load_Tmat(transform_path) for transform_path in handle_trans ])
	Tmat = numpy.swapaxes(Tmat,0,1)
	Tmat = Tmat.reshape(Tmat.shape[0], Tmat.shape[1]*Tmat.shape[2])
	
	weight_path = args.weights
	W = format_loader.load_DMAT( weight_path ).T

	output_path = args.output
	print( "Saving ground truth results to:", output_path )
	format_loader.write_result(output_path, Tmat.round(6), W.round(6), 0, 0, col_major=True)
	
