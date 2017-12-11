# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import time
import sys
import json
import scipy.sparse
import scipy.optimize
from trimesh import TriMesh
from autograd.numpy import *
import autograd.numpy as np
from autograd import elementwise_grad, jacobian



def load_DMAT( path ):
    from numpy import zeros
    
    with open( path ) as f:
        
        for i, line in enumerate( f ):
            if i == 0:
                dims = list( map( int, line.strip().split() ) )
                M = zeros( prod( dims ) )
            
            else:
                M[i-1] = float( line )
    
    M = M.reshape( dims )
    
    return M


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




def objective_func_vector(x0, vertices1, vertices2):
    ### vertices1 is N*4 (homogeneous), vertices2 is N*3. X0 is N*3*4
    N=len(vertices1)
    Matrix=x0.reshape((N, vertices2.shape[1], vertices1.shape[1]))

    reconstruct_vertices2=np.multiply(Matrix, vertices1.reshape((N,1,-1))).sum(axis=-1)

    return (reconstruct_vertices2-vertices2).reshape(-1)



def objective_func(x0, vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, W_spatial, control_level):
    
    obj=objective_func_vector(x0, vertices1, vertices2)
    val=np.square(obj).sum()

    M=vertices1.shape[1]*vertices2.shape[1]
    
    if W_svd!=0.0:
        Matrix=x0.reshape((-1,M))
        L=Matrix-Matrix.mean(axis=0).reshape((1,-1))
        s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
        
        
        # val+=np.square(s[-control_level:]).sum()*len(vertices1)*W_svd*0.001/control_level

            
        # val+=(np.maximum(s[-control_level:],1e-8)**0.5).sum()*len(vertices1)*W_svd*0.0001/control_level
        
        
        # val+=((-1.0)/(0.1+s[-control_level:])).sum()*len(vertices1)*W_svd*0.001/control_level
        

        val+=((-1.0)/(1+s[-control_level:]**2)).sum()*len(vertices1)*W_svd/control_level

        
        # val+=((-1.0)/(0.1+s[-control_level:]**2)).sum()*W_svd


        
        
    if W_rotation!=0.0:
        
        temp1=x0.reshape((-1,3,4))[:,:,:3]
        temp2=temp1.transpose((0,2,1))
        identities=np.repeat(np.identity(3).reshape((1,-1)), len(temp1), 0).ravel()
        val+=np.square((temp1[:,:,:,np.newaxis]*temp2[:,np.newaxis,:,:]).sum(axis=-2).ravel()-identities).sum()*W_rotation/M
        
        


    if W_spatial!=0.0:
        #### this is ok, but not supported by autograd library to compute gradient.
        val+=np.dot(x0,Smooth_Matrix.dot(x0))*W_spatial/M
    
    return val



def gradient_objective_func(x0, vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, W_spatial, control_level):

    grad=elementwise_grad(objective_func,0)
    g=grad(x0, vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, 0.0, control_level)
    
    M=vertices1.shape[1]*vertices2.shape[1]


    if W_spatial!=0.0:
        g2=2*Smooth_Matrix.dot(x0)*W_spatial/M
        g+=g2

    return g




def objective_func_and_gradient(x0, vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, W_spatial, control_level):
    obj=objective_func(x0, vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, W_spatial, control_level)
    grad=gradient_objective_func(x0, vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, W_spatial, control_level)
    return obj, grad
    
    


def optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level):


    start = time.clock()
    xlen = len( x0 )
    bounds=[(None,None)]*xlen
        
    W_svd=0.0
    W_sparse=0.0
    W_spatial=0.0
    
    if 'W_svd' in weights:
        W_svd=weights["W_svd"]
    if 'W_rotation' in weights:
        W_rotation=weights["W_rotation"]
    if 'W_spatial' in weights:
        W_spatial=weights["W_spatial"]

    res = scipy.optimize.minimize(objective_func, x0, args=(vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, W_spatial, control_level), jac = gradient_objective_func
            ,bounds = bounds
            # ,options={'gtol':1e-4, 'ftol': 1e-4}
            ,method='L-BFGS-B'
         )

    x=res["x"]
    print res["success"]
    end = time.clock()

    # print 'took ', (end-start), ' seconds.'

    return x



def optimize_basinhoppings(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level):

    start = time.clock()

    W_svd=0.0
    W_sparse=0.0
    W_spatial=0.0
    
    if 'W_svd' in weights:
        W_svd=weights["W_svd"]
    if 'W_rotation' in weights:
        W_rotation=weights["W_rotation"]
    if 'W_spatial' in weights:
        W_spatial=weights["W_spatial"]
        

    minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, 
                        "args": (vertices1, vertices2, Smooth_Matrix, W_svd, W_rotation, W_spatial, control_level) }
    
    res = scipy.optimize.basinhopping(objective_func_and_gradient, x0, minimizer_kwargs=minimizer_kwargs, 
                                      niter=20, 
                                      stepsize=0.1,
                                      T=2.0
                                     )

    x=res.x
    end = time.clock()

    # print 'took ', (end-start), ' seconds.'

    return x



def run_one(mesh1, mesh2, outprefix, weights, initials=None):

    vertices1=np.hstack((np.asarray(mesh1.vs),np.ones((len(mesh1.vs),1))))
    vertices2=np.asarray(mesh2.vs)
    M=vertices1.shape[1]*vertices2.shape[1]
    
    if initials is not None: #### assume it is numpy array
        x0=initials.copy()
    else:
        x0=np.ones(len(vertices1)*M)/M
        
        
    Smooth_Matrix=create_laplacian(mesh1, M)

    loop=0
    while loop<3:
        print "############################"
        print "loop: ", loop
        print weights

        for i in range(M):
            print "round: ", i+1
            x0_copy=x0.copy()
            ### directly on minimize() with L-BFGS-B
            transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

            ### basinhopping on L-BFGS-B
            # transformation_matrix=optimize_basinhoppings(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

            transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
            L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
            s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
            x0=transformation_matrix.ravel()
            print "singular values: ", s.round(3)
            rmse=np.square(objective_func_vector(x0, vertices1, vertices2)).sum()
            print "recontruction error: ", rmse
            if loop>=1 and rmse>1.0:
                x0=x0_copy.copy()
                break
        
        loop+=1
        # weights["W_svd"]-=0.5
        # weights["W_rotation"]-=0.05
        
    
    
    transformation_matrix=x0
    
    diff=objective_func_vector(transformation_matrix.ravel(), vertices1, vertices2)

    diff=np.square(diff)

    print 'max diff: ', sqrt(diff).max()
    print 'median diff', median(sqrt(diff))
    print 'RMSE: ', sqrt(diff.sum()/len(vertices1))
    
    return transformation_matrix.ravel()









    

    
    