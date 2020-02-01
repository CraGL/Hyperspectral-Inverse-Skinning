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
import scipy.optimize
import glob
import includes
import format_loader
import recover_poses

from trimesh import TriMesh

def match_data( gt_data, target ):
	assert( gt_data.shape == target.shape )
	
	ordering = []
	N = gt_data.shape[0]
	match_board=((gt_data.reshape((N,1,-1))-target.reshape((1,N,-1)))**2).sum(-1)
			
	row_ind, col_ind = scipy.optimize.linear_sum_assignment( match_board )
		
	return col_ind	

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser( description='Compare our results and SSD approch results with ground truth.' )
	parser.add_argument( 'rest_pose', type=str, help='Rest pose (OBJ).')
	parser.add_argument( 'pose_folder', type=str, help='Folder containing deformed poses.')
	parser.add_argument( 'our_result', type=str, help='our results(txt).')
	## UPDATE: type=bool does not do what we think it does. bool("False") == True.
	##		   For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	def str2bool(s): return {'true': True, 'false': False}[s.lower()]
	# parser.add_argument( '--write-OBJ', '-W', type=bool, default=False, help='whether to save recovered and SSD poses.')
	parser.add_argument( '--write-OBJ', '-W', type=str2bool, default=False, help='whether to save recovered and SSD poses.')
	parser.add_argument( '--debug', '-D', type=str2bool, default=False, help='print out debug information.')
	args = parser.parse_args()
	
	base_dir = args.pose_folder
	## mesh at rest pose
	gt_mesh_paths = glob.glob(base_dir + "/*.obj")
	gt_mesh_paths.sort()
	gt_vs = np.array([ TriMesh.FromOBJ_FileName(mesh_path).vs for mesh_path in gt_mesh_paths ])
		
	rest_mesh = TriMesh.FromOBJ_FileName(args.rest_pose)
	rest_vs = np.array(rest_mesh.vs)
	rest_fs = np.array(rest_mesh.faces)
	
	## assert name matches
	# assert( np.array([ os.path.split(os.path.basename(mesh_path))[0] == os.path.split(os.path.basename(bone_path))[0] for mesh_path, bone_path in zip(gt_mesh_paths, gt_bone_paths)]).all() )
	gt_names = [ os.path.basename(mesh_path) for mesh_path in gt_mesh_paths ]
	
	## diagonal
	diag = rest_vs.max( axis = 0 ) - rest_vs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	
	rev_bones_unordered, rev_w_unordered = format_loader.load_result(args.our_result)
	np.set_printoptions(precision=6, suppress=True)
	
	rev_bones = rev_bones_unordered.copy()
	rev_w = rev_w_unordered.copy()
	
	# col_ind = match_data(gt_bones, rev_bones)
	# rev_bones = np.array([ rev_bones[i] for i in col_ind ])
	# rev_w = np.array([ rev_w[i] for i in col_ind ])
	
	## Adjust bones data to Pose-by-bone-by-transformation
	# gt_bones = np.swapaxes(gt_bones, 0, 1) 
	rev_bones = np.swapaxes(rev_bones, 0, 1)
	rev_vs = np.array([recover_poses.lbs_all(rest_vs, rev_bones_per_pose, rev_w.T) for rev_bones_per_pose in rev_bones ])

	ordering = match_data( gt_vs, rev_vs )
	print( "match order of our recovery: ", ordering )
	rev_vs = np.array([ rev_vs[i] for i in ordering ])
	
#	if args.debug:
#		np.set_printoptions(threshold=np.nan)
#		print( "############################################" )
#		print( "Per-bone transformation P-by-B-by-12:" )
#		print( rev_bones )
#		print( "############################################" )
#		print( "weights N-by-B:" )
#		print( rev_w.T )
	
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
	
	
	N = len( rest_vs )
	print( "############################################" )
	print( "Reconstruction Mesh Error: " )
#	print( "rev error: ", np.linalg.norm(gt_vs - rev_vs)/(diag*N) )
	rev_error, rev_erms = compute_error(gt_vs, rev_vs)
	rev_error = rev_error / diag
	rev_erms = rev_erms *2 / diag
	print( "rev: max, mean and median per-vertex distance", np.max(rev_error), np.mean(rev_error), np.median(rev_error) )
	print( "Our E_RMS_kavan2010: ", rev_erms )
	print( "############################################" )
	
#	import pdb; pdb.set_trace()
	if args.write_OBJ:
		output_folder = os.path.split(args.our_result)[0]
		our_folder = output_folder + "/our_recovered"
		if not os.path.exists(our_folder):
			os.makedirs(our_folder)
			
		for i, vs in enumerate(rev_vs):
			output_path = os.path.join(our_folder, gt_names[i])
			format_loader.write_OBJ( output_path, vs.round(6), rest_fs )
	
