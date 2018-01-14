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

def plot(gt_bones, ssd_bones, rev_bones):
	########## visualize difference of recovered M and groundtruth M.
	import matplotlib.pyplot as plt
	import matplotlib.animation
	from mpl_toolkits.mplot3d import Axes3D

	assert( len(gt_bones) == len(ssd_bones) )
	
	fig= plt.figure()
	ax=fig.add_subplot(111, projection='3d')
	text = fig.suptitle('transformation matrix comparison')

	test_vals=np.array([
			[0,0,0,1],
			[1,0,0,1],
			[0,1,0,1],
			[0,0,1,1]
		])

	ssd_vals_list=[]
	gt_vals_list=[]
	rev_vals_list=[]

	ssd_bones=ssd_bones.reshape((-1, 12))
	rev_bones=rev_bones.reshape((-1, 12))
	gt_bones=gt_bones.reshape((-1, 12))
	
	frames=len(gt_bones)
	print( "# bones is ", frames )

	for num in range(frames):
		ssdi=ssd_bones[num]
		revi=rev_bones[num]
		gti=gt_bones[num]
		ssd_vals=np.multiply(ssdi.reshape((1,3,4)), test_vals.reshape((-1,1,4))).sum(axis=-1)
		rev_vals=np.multiply(revi.reshape((1,3,4)), test_vals.reshape((-1,1,4))).sum(axis=-1)
		gt_vals=np.multiply(gti.reshape((1,3,4)), test_vals.reshape((-1,1,4))).sum(axis=-1)
		ssd_vals_list.append(ssd_vals)
		gt_vals_list.append(gt_vals)
		rev_vals_list.append(rev_vals)

	def update_graph(num):
		ax.clear()##### if you want to show accumulated data in one figure, comment this line.
	
		ssd_vals=ssd_vals_list[num]
		gt_vals=gt_vals_list[num]
		rev_vals=rev_vals_list[num]
	
	
		for u1, v1, w1 in zip(test_vals[1:,0]-test_vals[0,0], test_vals[1:,1]-test_vals[0,1], test_vals[1:,2]-test_vals[0,2]):
			graph=ax.quiver(test_vals[0,0], test_vals[0,1], test_vals[0,2], u1, v1, w1, pivot = 'tail', length=np.sqrt(u1**2+v1**2+w1**2), color='r')
	
		for u1, v1, w1 in zip(ssd_vals[1:,0]-ssd_vals[0,0], ssd_vals[1:,1]-ssd_vals[0,1], ssd_vals[1:,2]-ssd_vals[0,2]):
			graph=ax.quiver(ssd_vals[0,0], ssd_vals[0,1], ssd_vals[0,2], u1, v1, w1, pivot = 'tail', length=np.sqrt(u1**2+v1**2+w1**2), color='g')
			
		for u1, v1, w1 in zip(rev_vals[1:,0]-rev_vals[0,0], rev_vals[1:,1]-rev_vals[0,1], rev_vals[1:,2]-rev_vals[0,2]):
			graph=ax.quiver(rev_vals[0,0], rev_vals[0,1], rev_vals[0,2], u1, v1, w1, pivot = 'tail', length=np.sqrt(u1**2+v1**2+w1**2), color='y')
	
		for u1, v1, w1 in zip(gt_vals[1:,0]-gt_vals[0,0], gt_vals[1:,1]-gt_vals[0,1], gt_vals[1:,2]-gt_vals[0,2]):
			graph=ax.quiver(gt_vals[0,0], gt_vals[0,1], gt_vals[0,2], u1, v1, w1, pivot = 'tail', length=np.sqrt(u1**2+v1**2+w1**2), color='b')

		lb=-2
		ub=2

		ax.set_xlim(lb,ub)
		ax.set_ylim(lb,ub)
		ax.set_zlim(lb,ub)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
	
		text.set_text('transformation matrix comparison={}'.format(num))
	
		return text, graph


	ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames, interval=1000)

	plt.show()

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
	parser.add_argument( 'weights', type=str, help='ground truth skinning weights.')
	parser.add_argument( 'our_result', type=str, help='our results(txt).')
	## UPDATE: type=bool does not do what we think it does. bool("False") == True.
	##         For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	def str2bool(s): return {'true': True, 'false': False}[s.lower()]
	# parser.add_argument( '--write-OBJ', '-W', type=bool, default=False, help='whether to save recovered and SSD poses.')
	parser.add_argument( '--write-OBJ', '-W', type=str2bool, default=False, help='whether to save recovered and SSD poses.')
	parser.add_argument( '--ssd_result', '--SSD', type=str, help='SSD results(txt).')
	parser.add_argument( '--debug', '-D', type=str2bool, default=False, help='print out debug information.')
	args = parser.parse_args()
	
	base_dir = args.pose_folder
	## mesh at rest pose
	gt_bone_paths = glob.glob(base_dir + "/*.Tmat")
	gt_bone_paths.sort()
	gt_bones = np.array([ format_loader.load_Tmat(transform_path) for transform_path in gt_bone_paths ])
	gt_bones = np.swapaxes(gt_bones, 0, 1)
	
	## Tmat are col-major, convert it to row-major
	gt_bones = gt_bones.reshape( gt_bones.shape[0], gt_bones.shape[1], 4, 3 )
	gt_bones = np.swapaxes(gt_bones, 2, 3)
	gt_bones = gt_bones.reshape( gt_bones.shape[0], gt_bones.shape[1], 12 )
		
	gt_mesh_paths = glob.glob(base_dir + "/*.obj")
	gt_mesh_paths.sort()
	gt_vs = np.array([ TriMesh.FromOBJ_FileName(mesh_path).vs for mesh_path in gt_mesh_paths ])
		
	rest_mesh = TriMesh.FromOBJ_FileName(args.rest_pose)
	rest_vs = np.array(rest_mesh.vs)
	rest_fs = np.array(rest_mesh.faces)
	
	## assert name matches
	assert( np.array([ os.path.split(os.path.basename(mesh_path))[0] == os.path.split(os.path.basename(bone_path))[0] for mesh_path, bone_path in zip(gt_mesh_paths, gt_bone_paths)]).all() )
	gt_names = [ os.path.basename(mesh_path) for mesh_path in gt_mesh_paths ]
	
	## diagonal
	diag = rest_vs.max( axis = 0 ) - rest_vs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	
	gt_w = format_loader.load_DMAT(args.weights)
		
	rev_bones_unordered, rev_w_unordered = format_loader.load_result(args.our_result)
	np.set_printoptions(precision=6, suppress=True)
	
	N = len(gt_bones)
	assert( len(rev_bones_unordered) == N )
	rev_bones = rev_bones_unordered.copy()
	rev_w = rev_w_unordered.copy()
	
	ssd_bones = np.zeros(rev_bones.shape)
	ssd_w = np.zeros(rev_w.shape)

	col_ind = match_data(gt_bones, rev_bones)
	rev_bones = np.array([ rev_bones[i] for i in col_ind ])
	rev_w = np.array([ rev_w[i] for i in col_ind ])
	
	if args.ssd_result is not None:
		ssd_bones, ssd_w = format_loader.load_result(args.ssd_result)
		col_ind = match_data(gt_bones, ssd_bones)
		ssd_bones = np.array([ ssd_bones[i] for i in col_ind ])
		ssd_w = np.array([ ssd_w[i] for i in col_ind ])
				
	## Adjust bones data to Pose-by-bone-by-transformation
	gt_bones = np.swapaxes(gt_bones, 0, 1) 
	rev_bones = np.swapaxes(rev_bones, 0, 1)
	rev_vs = np.array([recover_poses.lbs_all(rest_vs, rev_bones_per_pose, rev_w.T) for rev_bones_per_pose in rev_bones ])
	ssd_vs = np.zeros(rev_vs.shape)
	if args.ssd_result is not None:
		ssd_bones = np.swapaxes(ssd_bones, 0, 1)
		ssd_vs = np.array([recover_poses.lbs_all(rest_vs, ssd_bones_per_pose, ssd_w.T) for ssd_bones_per_pose in ssd_bones ])
	
# 	if args.debug:
# 		np.set_printoptions(threshold=np.nan)
# 		print( "############################################" )
# 		print( "Per-bone transformation P-by-B-by-12:" )
# 		print( rev_bones )
# 		print( "############################################" )
# 		print( "weights N-by-B:" )
# 		print( rev_w.T )
	
	def compute_error( gt, data ):
		error = []
		for pose_gt, pose_data in zip(gt, data):
			error.append( np.array([np.linalg.norm(pt_gt - pt_data) for pt_gt, pt_data in zip(pose_gt, pose_data)]) )
			
		return np.array(error)
	
	N = len( rest_vs )
	print( "############################################" )
	print( "Per-bone transformation Error: " )
	if args.ssd_result is not None:
		print( "ssd error: ", np.linalg.norm(gt_bones - ssd_bones)/gt_bones.size )
	print( "rev error: ", np.linalg.norm(gt_bones - rev_bones)/gt_bones.size )
	print( "############################################" )
	print( "Weight Error: " )
	if args.ssd_result is not None:
		print( "ssd error: ", np.linalg.norm(gt_w - ssd_w)/gt_w.size )
	print( "rev error: ", np.linalg.norm(gt_w - rev_w)/gt_w.size )
	print( "############################################" )
	print( "Reconstruction Mesh Error: " )
	if args.ssd_result is not None:
# 		print( "ssd error: ", np.linalg.norm(gt_vs - ssd_vs)/(diag*N) )
		ssd_error = compute_error(gt_vs, ssd_vs)/diag
		print( "ssd: max, mean and median per-vertex distance", np.max(ssd_error), np.mean(ssd_error), np.median(ssd_error) )
# 	print( "rev error: ", np.linalg.norm(gt_vs - rev_vs)/(diag*N) )
	rev_error = compute_error(gt_vs, rev_vs)/diag
	print( "rev: max, mean and median per-vertex distance", np.max(rev_error), np.mean(rev_error), np.median(rev_error) )
	print( "############################################" )
	
# 	import pdb; pdb.set_trace()
	if args.write_OBJ:
		output_folder = os.path.split(args.our_result)[0]
		our_folder = output_folder + "/our_recovered"
		if not os.path.exists(our_folder):
			os.makedirs(our_folder)
			
		for i, vs in enumerate(rev_vs):
			output_path = os.path.join(our_folder, gt_names[i])
			format_loader.write_OBJ( output_path, vs.round(6), rest_fs )
		
		if args.ssd_result is not None:
			ssd_folder = output_folder + "/ssd_recovered"
			if not os.path.exists(ssd_folder):
				os.makedirs(ssd_folder)
			
			for i, vs in enumerate(ssd_vs):
				output_path = os.path.join(ssd_folder, gt_names[i])
				format_loader.write_OBJ( output_path, vs.round(6), rest_fs )
			
# 	plot(gt_bones, ssd_bones, rev_bones )	
 
	
	
	
		
