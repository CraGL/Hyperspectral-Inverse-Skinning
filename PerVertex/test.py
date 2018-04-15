import numpy as np
from local_subspace_recover import *
import glob
from trimesh import TriMesh
from format_loader import load_DMAT
##### verify using cheburashka example to see if flat_intersection.py load the qs data correctly
##### here we project qs to 10 dimenstion and recover to original dimension, and then compute T error and V error.


# #### read obj file
# name="cheburashka"
# handles=11 ### groundtruth handles
name="cylinder"
handles=4

sampled_subset_size=48
search_range=120
sample_method="none"
method="vertex"
version=0
svd_threshold=1e-15
transformation_percentile=100


base_dir="../models/"+name+"/"
filenames=glob.glob(base_dir+"*.obj")
mesh0=TriMesh.FromOBJ_FileName("../models/"+name+".obj")
print filenames
mesh1_list=[]
for i in range(len(filenames)):
    mesh1_list.append(TriMesh.FromOBJ_FileName(filenames[i]))

groundtruth_names=glob.glob(base_dir+"*.DMAT")
print groundtruth_names

gt_all=None
for i in range(len(groundtruth_names)):
    gt=load_DMAT(groundtruth_names[i])
    gt=gt.T
    gt=gt.reshape((-1,4,3))
    gt=np.transpose(gt, (0,2,1)).reshape((-1,12))
    if gt_all is None:
        gt_all=gt
    else:
        gt_all=np.hstack((gt_all, gt))
    

print gt_all.shape


vertices0=vertices0_temp=np.hstack((np.asarray(mesh0.vs),np.ones((len(mesh0.vs),1))))
vertices1=vertices1_temp=np.asarray(mesh1_list[0].vs)
for i in range(1,len(mesh1_list)):
    vertices0=np.hstack((vertices0, vertices0_temp))
    vertices1=np.hstack((vertices1, np.asarray(mesh1_list[i].vs)))
print vertices0.shape
print vertices1.shape

pose_num=len(mesh1_list)


qs, errors, smallest_singular_values = find_subspace_intersections( mesh0, mesh1_list, version, method = method, random_sample_close_vertex=sample_method, candidate_num=search_range, vertices_num=sampled_subset_size)
print qs

qs=np.loadtxt("/Users/jianchao/InverseSkinning/results_clean_vertex_0_copy/cylinder-4/local_subspace_recover_none.txt")
print qs

scale=find_scale(vertices0[:,:3])
print 1/(scale*len(qs))
print vertex_reconstruction_error(vertices0, vertices1, qs, scale=scale)
print transformation_matrix_error(gt_all, qs)


qs_to_save = qs[ smallest_singular_values>=svd_threshold ]
topN = int( len( qs_to_save )*transformation_percentile/100.0 )
indices=np.argsort( errors[ smallest_singular_values>=svd_threshold ] )[:topN]
indices=np.arange(len(qs))
qs_to_save = qs_to_save[ indices , : ]
print( "About to save qs with dimension:", qs_to_save.shape )
gt_to_save=gt_all[indices,:]
print qs_to_save
project_func, restore_func=uncorrellated_space(qs_to_save, H=handles-1)
restored_data=restore_func(project_func(qs_to_save))
print vertex_reconstruction_error(vertices0[indices,:], vertices1[indices,:], restored_data, scale=scale)
print transformation_matrix_error(gt_to_save, restored_data)
