from __future__ import print_function, division
from recordclass import recordclass

import numpy as np
import json
import time
import scipy
import scipy.sparse
from trimesh import TriMesh
import glob

def compare_one_pose(mesh0, mesh1, groundtruth=None):
	output_prefix = "Vertex_tramsformation_matrix_recovering"
	weights = {'W_svd': 2.0, 'W_rotation': 0.01, 'W_spatial': 0.0 }

	start=time.time()

	print( "solving transformation matrix" )

	vertices0=np.hstack((np.asarray(mesh0.vs),np.ones((len(mesh0.vs),1))))
	vertices1=np.asarray(mesh1.vs)
	M=vertices0.shape[1]*vertices1.shape[1]

	x0=np.ones(len(vertices1)*M)/M
	
	res=run_one(mesh0, mesh1, output_prefix, weights, x0)
	res=res.reshape((len(mesh0.vs),-1))
	end=time.time()
	print( "using time: ", end-start )
	print( "result's shape" )
	
	import pdb;pdb.set_trace()
	print( res.shape )
	
	assert( len(groundtruth) == 2 and groundtruth.shape[1] == 12 )
	gt = groundtruth.reshape((-1,4,3))
	gt = np.transpose(gt, (0,2,1)).reshape((-1,12))
	
	diff = abs(res - gt)
	print( "Largest error: ", max( diff ) )
	
	vertices0=np.hstack((np.asarray(mesh0.vs),np.ones((len(mesh0.vs),1))))
	print( "vertices reconstruction error: " )
	print( np.multiply(res.reshape((-1,3,4)), vertices0.reshape((-1,1,4))).sum(axis=-1)[:len(ind)]-asarray(mesh1.vs)[:len(ind)] )
	
	L = res-res.mean(axis=0).reshape((1,-1))
	eigenVal,eigenVec=np.linalg.eig(L.T.dot(L))
	print( sorted(eigenVal) )
	
	gtL=gt-gt.mean(axis=0).reshape((1,-1))
	gteigenVal,gteigenVec=np.linalg.eig(gtL.T.dot(gtL))
	print( sorted(gteigenVal) )

	L=res-res.mean(axis=0).reshape((1,-1))
	s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
	print( s.round(3) )
	
	gtL=gt-gt.mean(axis=0).reshape((1,-1))
	s = np.linalg.svd(gtL, full_matrices=True, compute_uv=False)
	print( s.round(3) )
 	
	temp1=res.reshape((-1,3,4))[:,:,:3]
	temp2=temp1.transpose((0,2,1))
	identities=np.repeat(np.identity(3).reshape((1,-1)), len(temp1), 0).ravel()
	print abs((temp1[:,:,:,np.newaxis]*temp2[:,np.newaxis,:,:]).sum(axis=-2).ravel()-identities).max()
	
	temp1=gt.reshape((-1,3,4))[:,:,:3]
	temp2=temp1.transpose((0,2,1))
	identities=np.repeat(np.identity(3).reshape((1,-1)), len(temp1), 0).ravel()
	print abs((temp1[:,:,:,np.newaxis]*temp2[:,np.newaxis,:,:]).sum(axis=-2).ravel()-identities).max()	

if __name__ == '__main__':
	np.set_printoptions(linewidth=2000, suppress=True)
	from Extract_Transformation_matrix_minimize_SVD_singular_value import *

	#### read obj file
	import sys
	if len( sys.argv ) != 4 and len( sys.argv ) != 3:
		print( "usage: ", sys.argv[0], "path/to/rest_pose, path/to/deformed_pose[, path/to/groundtruth_weights]" )
		exit(-1)
		
	rest_pose = TriMesh.FromOBJ_FileName( sys.argv[1] )
	deformed_pose = TriMesh.FromOBJ_FileName( sys.argv[2] )
	groundtruth_weights = None
	if len( sys.argv ) == 4:	
		groundtruth_weights = load_DMAT(sys.argv[3]).T
		
	compare_one_pose( rest_pose, deformed_pose, groundtruth_weights )
	
	