"""
Compute E_RMS_kavan2010.
Adapted from compute.py by Songrun Liu.
"""

from __future__ import print_function, division

import os
import argparse
import numpy as np
import scipy.optimize
import glob

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
	parser.add_argument( 'truth_folder', type=str, help='Folder containing ground truth deformed meshes.')
	parser.add_argument( 'test_folder', type=str, help='Folder containing deformed meshes for comparison.')
	## UPDATE: type=bool does not do what we think it does. bool("False") == True.
	##		   For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	def str2bool(s): return {'true': True, 'false': False}[s.lower()]
	# parser.add_argument( '--debug', '-D', type=str2bool, default=False, help='print out debug information.')
	args = parser.parse_args()
	
	print( "=== %s ===" % args.test_folder )
	
	rest_mesh = TriMesh.FromOBJ_FileName(args.rest_pose)
	rest_vs = np.array(rest_mesh.vs)
	rest_fs = np.array(rest_mesh.faces)
	
	gt_mesh_paths = glob.glob(os.path.join( args.truth_folder, "*.obj" ))
	gt_mesh_paths.sort()
	gt_vs = np.array([ TriMesh.FromOBJ_FileName(mesh_path).vs for mesh_path in gt_mesh_paths ])
	
	test_mesh_paths = glob.glob(os.path.join( args.test_folder, "*.obj" ))
	test_mesh_paths.sort()
	test_vs = np.array([ TriMesh.FromOBJ_FileName(mesh_path).vs for mesh_path in test_mesh_paths ])
	
	## assert name matches
	gt_names = [ os.path.basename(mesh_path) for mesh_path in gt_mesh_paths ]
	test_names = [ os.path.basename(mesh_path) for mesh_path in test_mesh_paths ]
	# assert tuple(gt_names) == tuple(test_names)
	print( "gt_names:" )
	print( gt_names )
	print( "test_names:" )
	print( test_names )
	
	## diagonal
	diag = rest_vs.max( axis = 0 ) - rest_vs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	
	np.set_printoptions(precision=6, suppress=True)
	
	ordering = match_data( gt_vs, test_vs )
	print( "match order of our recovery: ", ordering )
	test_vs = np.array([ test_vs[i] for i in ordering ])
	
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
#	print( "rev error: ", np.linalg.norm(gt_vs - test_vs)/(diag*N) )
	test_error, test_erms = compute_error(gt_vs, test_vs)
	test_error = test_error / diag
	test_erms = test_erms *2 / diag
	print( "rev: max, mean and median per-vertex distance", np.max(test_error), np.mean(test_error), np.median(test_error) )
	print( "E_RMS_kavan2010: ", test_erms )
	print( "############################################" )
