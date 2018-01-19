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
import scipy.optimize

from trimesh import TriMesh
	
def lbs_all(rest, bones, W, scale=None):
	'''
	bones are a list of flattened row-major matrices
	scale has shape of B-by-3
	'''
	assert( len(bones.shape) == 2 )
	assert( len(W.shape) == 2 )
	assert( bones.shape[0] == W.shape[1] )
	assert( bones.shape[1] == 12 )
	assert( W.shape[0] == rest.shape[0] )
	assert( rest.shape[1] == 3 )
	
	if scale is not None:
		assert( len(scale.shape) == 2)
		assert( scale.shape[0] == bones.shape[0] )
		assert( scale.shape[1] == 3 )
	
	per_vertex = np.dot(W, bones)
	N = rest.shape[0]
	result = np.zeros((N,3))
	
	for i in range(N):
		tran = per_vertex[i].reshape(3,4)
		p = rest[i]
		result[i] = np.dot(tran[:, :3], p[:,np.newaxis]).squeeze() + tran[:, 3]
		if scale is not None:
			pass
		
	return result

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
	parser.add_argument( 'result', type=str, help='our results(txt).')
	parser.add_argument( '--ssd_result', '--SSD', type=str, help='SSD results(txt).')
	parser.add_argument( '--output', '-O', type=str, help='path to save recovered poses.')
	parser.add_argument( '--showAll', '--all', type=bool, default=False, help='print the error for each pose')
	args = parser.parse_args()
		
	rest_mesh = TriMesh.FromOBJ_FileName(args.rest_pose)
	rest_vs = np.array(rest_mesh.vs)
	rest_fs = np.array(rest_mesh.faces)
		
	rev_bones, rev_w = format_loader.load_result(args.result)
	np.set_printoptions(precision=6, suppress=True)
				
	## Adjust bones data to Pose-by-bone-by-transformation
	rev_bones = np.swapaxes(rev_bones, 0, 1)
	rev_vs = np.array([lbs_all(rest_vs, rev_bones_per_pose, rev_w.T) for rev_bones_per_pose in rev_bones ])	
	
	gt_mesh_paths = glob.glob(args.pose_folder + "/*.obj")
	gt_mesh_paths.sort()
	gt_vs = np.array([ TriMesh.FromOBJ_FileName(mesh_path).vs for mesh_path in gt_mesh_paths ])
	gt_names = [os.path.basename(mesh_path) for mesh_path in gt_mesh_paths]
	
	## diagonal
	diag = rest_vs.max( axis = 0 ) - rest_vs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	N = len(rest_vs)
	P = len( gt_vs )
	
	ordering = match_data( gt_vs, rev_vs )
	print( "match order of our recovery: ", ordering )
	rev_vs = np.array([ rev_vs[i] for i in ordering ])
	
	if args.output != "NO":
		output_folder = os.path.split(args.result)[0]		
		if args.output is not None:
			output_folder = args.output
		
		our_folder = output_folder + "/our_recovered"
		if not os.path.exists(our_folder):
			os.makedirs(our_folder)
		
		for i, vs in enumerate(rev_vs):
			output_path = os.path.join(our_folder, gt_names[i])
			format_loader.write_OBJ( output_path, vs.round(6), rest_fs )
		
	def compute_error( gt, data ):
		error = []
		for pose_gt, pose_data in zip(gt, data):
			error.append( np.array([np.linalg.norm(pt_gt - pt_data) for pt_gt, pt_data in zip(pose_gt, pose_data)]) )
		
		## Divide this by the bounding sphere radius (or 1/2 the bounding box diagonal?)
		## to get the error metric E_RMS from Kavan 2010.
		E_RMS_kavan2010 = 1000*np.linalg.norm( gt.ravel() - data.ravel() )/np.sqrt(3*gt.shape[0]*gt.shape[1])
		
		return np.array(error), E_RMS_kavan2010
		
	print( "############################################" )
	print( os.path.basename(args.rest_pose), rev_w.shape[0], "handles" )
	print( "Reconstruction Mesh Error: " )
	# print( "rev error: ", np.linalg.norm(gt_vs - rev_vs)/(diag*N) )
	rev_error, rev_erms = compute_error(gt_vs, rev_vs)
	rev_error = rev_error / diag
	rev_erms = rev_erms *2 / diag
	print( "rev: max, mean and median per-vertex distance", np.max(rev_error), np.mean(rev_error), np.median(rev_error) )
	print( "Our E_RMS_kavan2010: ", rev_erms )
	if args.showAll:
		print( "rev per pose max, mean and median error:")
		for i in range( len(rev_error) ):
			print( gt_names[i], np.max(rev_error[i]), np.mean(rev_error[i]), np.median(rev_error[i]) )
	if args.ssd_result is not None:
		ssd_bones, ssd_w = format_loader.load_result(args.ssd_result)
		ssd_bones = np.swapaxes(ssd_bones, 0, 1)
		ssd_vs = np.array([lbs_all(rest_vs, ssd_bones_per_pose, ssd_w.T) for ssd_bones_per_pose in ssd_bones ])
		
		ordering = match_data( gt_vs, ssd_vs )
		print( "match order of ssd recovery: ", ordering )
		ssd_vs = np.array([ ssd_vs[i] for i in ordering ])
		
		# print( "ssd error: ", np.linalg.norm(gt_vs - ssd_vs)/(diag*N) )
		ssd_error, ssd_erms = compute_error(gt_vs, ssd_vs)
		ssd_error = ssd_error / diag
		ssd_erms = ssd_erms *2 / diag
		print( "ssd: max, mean and median per-vertex distance", np.max(ssd_error), np.mean(ssd_error), np.median(ssd_error) )
		print( "SSD E_RMS_kavan2010: ", ssd_erms )
		if args.showAll:
			print( "ssd per pose max, mean and median error:")
			for i in range( len(ssd_error) ):
				print( gt_names[i], np.max(ssd_error[i]), np.mean(ssd_error[i]), np.median(ssd_error[i]) )

		ssd_folder = output_folder + "/ssd_recovered"
		if not os.path.exists(ssd_folder):
			os.makedirs(ssd_folder)
		
		for i, vs in enumerate(ssd_vs):
			output_path = os.path.join(ssd_folder, gt_names[i])
			format_loader.write_OBJ( output_path, vs.round(6), rest_fs )
	print( "############################################" )


	
 
	
	
	
		
