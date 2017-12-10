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
import includes
import format_loader

from trimesh import TriMesh
from space_mapper import SpaceMapper
from PerVertex.Extract_Transformation_matrix_minimize_SVD_singular_value_allPoses import *

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
	
	
def lbs_all(rest, bones, W):
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
		tran = per_vertex[i].reshape(4,3).T
		p = rest[i]
		result[i] = np.dot(tran[:, :3], p[:,np.newaxis]).squeeze() + tran[:, 3]
	
# 	import pdb; pdb.set_trace()
	
	return result
	

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print( 'Usage:', sys.argv[0], 'path/to/groundtruth', 'path/to/gt_rest_pose', 'path/to/gt_weights', 'path/to/poses.txt', file = sys.stderr )
		exit(-1)
	
	base_dir = sys.argv[1]
	## mesh at rest pose
	gt_bone_paths = glob.glob(base_dir + "/*.Tmat")
	gt_bone_paths.sort()
	gt_bones = np.array([ format_loader.load_Tmat(transform_path).T for transform_path in gt_bone_paths ])
	gt_bones = np.swapaxes(gt_bones, 0, 1)
	
	gt_mesh_paths = glob.glob(base_dir + "/*.obj")
	gt_mesh_paths.sort()
	gt_vs = np.array([ TriMesh.FromOBJ_FileName(mesh_path).vs for mesh_path in gt_mesh_paths ])
	
	recovered_path = base_dir + "/result.txt"
	rev_bones_unordered, rev_w_unordered = format_loader.load_result(recovered_path)
	
	rest_mesh = TriMesh.FromOBJ_FileName(sys.argv[2])
	rest_vs = np.array(rest_mesh.vs)
	gt_w = format_loader.load_DMAT(sys.argv[3])
	
	ssd_bones_unordered, ssd_w_unordered = format_loader.load_result(sys.argv[4])
	np.set_printoptions(precision=6, suppress=True)
	
	N = len(gt_bones)
	assert( len(rev_bones_unordered) == N and len(ssd_bones_unordered) == N )
	ssd_bones = ssd_bones_unordered.copy()
	rev_bones = rev_bones_unordered.copy()
	ssd_w = ssd_w_unordered.copy()
	rev_w = rev_w_unordered.copy()
	
		
	for i, gt_vals in enumerate(gt_bones):
	
		rev_dist, ssd_dist = inf, inf
		rev_idx, ssd_idx = -1, -1
	
		for j in range(i, N):
			if( np.linalg.norm( gt_vals - rev_bones_unordered[j] ) < rev_dist ):
				rev_idx = j
				rev_dist = np.linalg.norm( gt_vals - rev_bones_unordered[j] )
			
			if( np.linalg.norm( gt_vals - ssd_bones_unordered[j] ) < ssd_dist ):
				ssd_idx = j
				ssd_dist = np.linalg.norm( gt_vals - ssd_bones_unordered[j] )
		
		rev_bones[i] = rev_bones_unordered[rev_idx]
		rev_bones_unordered[rev_idx] = rev_bones_unordered[i]
		
		rev_w[i] = rev_w_unordered[rev_idx]
		rev_w_unordered[rev_idx] = rev_w_unordered[i]
		
		ssd_bones[i] = ssd_bones_unordered[ssd_idx]
		ssd_bones_unordered[ssd_idx] = ssd_bones_unordered[i]
		
		ssd_w[i] = ssd_w_unordered[ssd_idx]
		ssd_w_unordered[ssd_idx] = ssd_w_unordered[i]
		
# 	gt_vs = gt_vs.reshape(len(gt_vs), -1)
	
	## Adjust bones data to Pose-by-bone-by-transformation
	gt_bones = np.swapaxes(gt_bones, 0, 1) 
	ssd_bones = np.swapaxes(ssd_bones, 0, 1) 
	rev_bones = np.swapaxes(rev_bones, 0, 1) 

	ssd_vs = np.array([lbs_all(rest_vs, ssd_bones_per_pose, ssd_w.T) for ssd_bones_per_pose in ssd_bones ])
	rev_vs = np.array([lbs_all(rest_vs, rev_bones_per_pose, rev_w.T) for rev_bones_per_pose in rev_bones ])
			
	print( "############################################" )
	print( "Pre-bone transformation Error: " )
	print( "ssd error: ", linalg.norm(gt_bones - ssd_bones) )
	print( "rev error: ", linalg.norm(gt_bones - rev_bones) )
	print( "############################################" )
	print( "Weight Error: " )
	print( "ssd error: ", linalg.norm(gt_w - ssd_w) )
	print( "rev error: ", linalg.norm(gt_w - rev_w) )
	print( "############################################" )
	print( "Reconstruction Mesh Error: " )
	print( "ssd error: ", linalg.norm(gt_vs - ssd_vs) )
	print( "rev error: ", linalg.norm(gt_vs - rev_vs) )
	print( "############################################" )

	name = sys.argv[2].split("/")[-1]
	gt_fs = rest_mesh.faces
	def write_OBJ( path, vs, fs ):
		with open( path, 'w' ) as file:
			for v in vs:
				file.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")	
			for f in fs:
				file.write("f " + str(f[0]+1) + " " + str(f[1]+1) + " " + str(f[2]+1) + "\n")
	
# 	import pdb; pdb.set_trace()
	for i, vs in enumerate(ssd_vs):
		output_path = "SSD_res/recon_ssd_P"+str(i)+"_"+name
		write_OBJ( output_path, vs.round(6), gt_fs )
	
	for i, vs in enumerate(rev_vs):
		output_path = "SSD_res/recon_rev_P"+str(i)+"_"+name
		write_OBJ( output_path, vs.round(6), gt_fs )
		
# 	plot(gt_bones, ssd_bones, rev_bones )	
 
	
	
	
		
