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
	

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print( 'Usage:', sys.argv[0], 'path/to/groundtruth', 'path/to/poses.txt', file = sys.stderr )
		exit(-1)
	
	base_dir = sys.argv[1]
	## mesh at rest pose
	gt = glob.glob(base_dir + "/*.Tmat")
	gt_bones = np.array([ format_loader.load_Tmat(transform_path).T for transform_path in gt ])
	gt_bones = np.swapaxes(gt_bones, 0, 1)
	
	recovered_path = base_dir + "/result.txt"
	rev_bones = format_loader.load_ssd_result(recovered_path)
	
	ssd_bones = format_loader.load_ssd_result(sys.argv[2])
	np.set_printoptions(precision=6, suppress=True)
	
	print( "original ssd result:")
	print( ssd_bones )
	N = len(gt_bones)
	assert( len(ssd_bones) == N and len(rev_bones) == N )
	rev_dist, ssd_dist = inf, inf
	rev_idx, ssd_idx = -1, -1
	ssd_res = ssd_bones.copy()
	rev_res = rev_bones.copy()
	
	for i, gt_vals in enumerate(gt_bones):
		for j in range(i, N):
			if( np.linalg.norm( gt_vals - rev_bones[j] ) < rev_dist ):
				rev_idx = j
				rev_dist = np.linalg.norm( gt_vals - rev_bones[j] )
			
			if( np.linalg.norm( gt_vals - ssd_bones[j] ) < ssd_dist ):
				ssd_idx = j
				ssd_dist = np.linalg.norm( gt_vals - ssd_bones[j] )
		
		rev_res[i] = rev_bones[rev_idx]
		rev_bones[rev_idx] = rev_bones[i]
		
		ssd_res[i] = ssd_bones[ssd_idx]
		ssd_bones[ssd_idx] = ssd_bones[i]
		
		rev_dist, ssd_dist = inf, inf
		rev_idx, ssd_idx = -1, -1	
			
	
	print( "ssd error: ", linalg.norm(gt_bones - ssd_res) )
	print( "rev error: ", linalg.norm(gt_bones - rev_res) )
	
# 	import pdb; pdb.set_trace()
		
	plot(gt_bones, ssd_res, rev_res )	
 
	
	
	
		
