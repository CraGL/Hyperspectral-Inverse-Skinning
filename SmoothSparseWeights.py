# -*- coding: utf-8 -*-
from __future__ import print_function, division

import time
import sys
import json
import scipy.sparse
import scipy.optimize
from trimesh import TriMesh
from format_loader import *
from autograd.numpy import *
import autograd.numpy as np
from autograd import elementwise_grad, jacobian
import glob


def Get_basic_Laplacian_sparse_matrix(mesh):
    n = len(mesh.vs) # N x 3
    I = []
    J = []
    V = []

    # Build sparse Laplacian Matrix coordinates and values
    for i in range(n):
        indices = mesh.vertex_vertex_neighbors(i)
        z = len(indices)
        I = I + ([i] * (z + 1)) # repeated row
        J = J + indices + [i] # column indices and this row
        V = V + ([-1] * z) + [z] # negative weights and row degree

    L = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n)).tocsr()
    return L



def create_laplacian(mesh, M):
    Lap = Get_basic_Laplacian_sparse_matrix(mesh)
    ## Now repeat Lap #pigments times.
    ## Because the layer values are the innermost dimension,
    ## every entry (i,j, val) in Lap should be repeated
    ## (i*#pigments + k, j*#pigments + k, val) for k in range(#pigments).
    Lap = Lap.tocoo()
    ## Store the shape. It's a good habit, because there may not be a nonzero
    ## element in the last row and column.
    shape = Lap.shape

    ## Fastest
    ks = arange( M )
    rows = ( repeat( asarray( Lap.row ).reshape( Lap.nnz, 1 ) * M, M, 1 ) + ks ).ravel()
    cols = ( repeat( asarray( Lap.col ).reshape( Lap.nnz, 1 ) * M, M, 1 ) + ks ).ravel()
    vals = ( repeat( asarray( Lap.data ).reshape( Lap.nnz, 1 ), M, 1 ) ).ravel()

    Lap = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( shape[0]*M, shape[1]*M ) ).tocsr()
    return Lap


def recover_vertices(x0, vertices1, endmembers):
    N=len(vertices1)
    Matrix=np.dot(x0,endmembers)
    Matrix=Matrix.reshape((N, -1, 3, 4))
    reconstruct_vertices2=np.multiply(Matrix, vertices1.reshape((N, -1, 1, 4))).sum(axis=-1)
    return reconstruct_vertices2


def objective_func_vector(x0, vertices1, vertices2, endmembers, fixed_x0, fixed_x1, choice=0):
    ### vertices1 is N*P*4 (homogeneous), vertices2 is N*P*3. X0 is N*handle, P is pose number 
    ## endmemebers is handle* 12p?
    # print ("objective function choice: ", choice)

    N=len(vertices2)
    H=len(x0)//len(vertices2)
    P=vertices2.shape[1]//3
    x0=x0.reshape((-1, H))
    endmembers=endmembers.reshape((H,-1))
    
    if choice==0:
        reconstruct_vertices2=recover_vertices(x0, vertices1, endmembers)
        return (reconstruct_vertices2-vertices2.reshape((N,-1,3))).reshape(-1)
    
    elif choice>=1: 

        fixed_x0=fixed_x0.reshape((-1,H))
        Matrix=np.dot(x0,endmembers)
        Matrix2=np.dot(fixed_x0,endmembers)
        return Matrix.ravel()-Matrix2.ravel()



def objective_func(x0, vertices1, vertices2, Smooth_Matrix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, choice=0):

    H=len(x0)//len(vertices2)
    P=vertices2.shape[1]//3

    obj=objective_func_vector(x0, vertices1, vertices2, endmembers, fixed_x0, fixed_x1, choice=choice)

    val=np.square(obj).sum()/P
    # print (val/len(vertices1))

    
    W_spatial=0.0
    W_sum=0.0
    W_sparse=0.0

    if 'W_spatial' in weights:
        W_spatial=weights["W_spatial"]
    if 'W_sum' in weights:
        W_sum=weights["W_sum"]
    if 'W_sparse' in weights:
        W_sparse=weights["W_sparse"]
    
    if W_sum!=0.0:
        temp=x0.reshape((-1, H))
        val+=np.square(temp.sum(axis=-1)-1.0).sum()*W_sum

    if W_sparse!=0.0:
        val+=(1.0-np.square(x0-1.0)).sum()*W_sparse/H

    if W_spatial!=0.0:
        #### this is not supported by autograd library to compute gradient.
        if choice<=1:
            val+=np.dot(x0-fixed_x0,Smooth_Matrix.dot(x0-fixed_x0))*W_spatial/H
        elif choice==2:
            val+=np.dot(x0-fixed_x1,Smooth_Matrix.dot(x0-fixed_x1))*W_spatial/H

    return val



def gradient_objective_func(x0, vertices1, vertices2, Smooth_Matrix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, choice=0):

    W_spatial=weights['W_spatial']
    weights['W_spatial']=0.0 #### turn off, because it seems autograd cannot support scipy.sparse matrix's dot product.
    grad=elementwise_grad(objective_func,0)
    g=grad(x0, vertices1, vertices2, Smooth_Matrix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, choice=choice)
    weights['W_spatial']=W_spatial ### recover
    
    H=len(x0)//len(vertices2)

    if W_spatial!=0.0:
        if choice<=1:
            g2=2*Smooth_Matrix.dot(x0-fixed_x0)*W_spatial/H
        elif choice==2:
            g2=2*Smooth_Matrix.dot(x0-fixed_x1)*W_spatial/H

        g+=g2
    

    if len(grad_zero_indices)!=0:
        g[grad_zero_indices]=0.0

    return g



def optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, choice=0):


    start = time.clock()
    bounds=[(0,1)]*len(x0)

    res = scipy.optimize.minimize(objective_func, x0, args=(vertices1, vertices2, Smooth_Matrix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, choice)
            ,jac = gradient_objective_func
            # ,options={'gtol':1e-4, 'ftol': 1e-4}
            ,method='L-BFGS-B'
            ,bounds=bounds
                                  
         )

    x=res["x"]
    print( res["success"] )
    end = time.clock()
    print (res["fun"])

    # print 'took ', (end-start), ' seconds.'

    return x






def run(mesh1, mesh2_list, outprefix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, initials=None, choice=0):
    
    vertices1=vertices1_temp=np.hstack((np.asarray(mesh1.vs),np.ones((len(mesh1.vs),1))))
    vertices2=vertices2_temp=np.asarray(mesh2_list[0].vs)
    for i in range(1,len(mesh2_list)):
        vertices1=np.hstack((vertices1, vertices1_temp))
        vertices2=np.hstack((vertices2, np.asarray(mesh2_list[i].vs)))
    print( vertices1.shape )
    print( vertices2.shape )

    ## scale the vertices.
    scale=find_scale(vertices1)/2
    vertices1/=scale
    vertices2/=scale

    H=len(endmembers)


    if initials is not None: #### assume it is numpy array
        x0=initials.copy()
    else:
        x0=np.ones(len(vertices1)*H)/H


    Smooth_Matrix1=create_laplacian(mesh1, H)
    
    Smooth_Matrix=Smooth_Matrix1.T.dot(Smooth_Matrix1) ## bi-laplacian.
    
    x=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, choice=choice)
    
    return x


def clip_first_k_values(matrix, k):
    ### matrix shape is N*H
    indices=np.argsort(matrix, axis=1)
    output=matrix.copy()
    fixed_indices=[] ## record the positions that filled with zeros. Will fix these values when run optimization.
    if 0<=k and k<len(indices):
        for i in range(len(indices)):
            index=indices[i]
            output[i, index[:-k]]=0.0
            fixed_indices.append(index[:-k]+indices.shape[1]*i)
    return output, np.asarray(fixed_indices).ravel()

def find_scale(Vertices):
    Dmin=Vertices.min(axis=0)
    Dmax=Vertices.max(axis=0)
    D=Dmax-Dmin
    scale=np.sqrt(D.dot(D))
    return scale

def E_RMS_kavan2010( gt, data, scale=1.0):
    ### for vertex error
    E_RMS_kavan2010 = 1000*np.linalg.norm( gt.ravel() - data.ravel() )*2.0/np.sqrt(len(gt.ravel())*scale*scale) ## 3*pose_num*vs_num, and scale!
    return E_RMS_kavan2010
