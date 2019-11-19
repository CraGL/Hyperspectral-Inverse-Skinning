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
from SmoothSparseWeights import *


def objective_func_endmembers(x0, mixing_weights, vertices1, vertices2, fixed_points, choice):
    N=len(vertices2)
    P=vertices2.shape[1]//3
    H=len(x0)//(P*12)

    mixing_weights=mixing_weights.reshape((-1, H))
    x0=x0.reshape((H, -1))

    if choice==0:
        Matrix=np.dot(mixing_weights, x0)
        Matrix=Matrix.reshape((N, -1, 3, 4))
        reconstruct_vertices2=np.multiply(Matrix, vertices1.reshape((N, -1, 1, 4))).sum(axis=-1)
        obj= (reconstruct_vertices2-vertices2.reshape((N,-1,3)))
        return np.square(obj).sum()/P

    
    elif choice>=1:
        obj=np.dot(mixing_weights, x0)-fixed_points.reshape((N, -1))
        return np.square(obj).sum()/P



def gradient_objective_func_endmembers(x0, mixing_weights, vertices1, vertices2, fixed_points, choice):
    grad=elementwise_grad(objective_func_endmembers,0)
    g=grad(x0, mixing_weights, vertices1, vertices2, fixed_points, choice)

    return g


def optimize_endmembers(x0, vertices1, vertices2, mixing_weights, fixed_points, choice):

    start = time.clock()

    res = scipy.optimize.minimize(objective_func_endmembers, x0, args=(mixing_weights, vertices1, vertices2, fixed_points, choice)
            ,jac = gradient_objective_func_endmembers
            ,options={'gtol':1e-4, 'ftol': 1e-4}
            ,method='L-BFGS-B'                                  
         )

    x=res["x"]
    print( res["success"] )
    end = time.clock()

    # print 'took ', (end-start), ' seconds.'

    return x


def run(mesh1, mesh2_list, outprefix, weights, endmembers, fixed_x0, fixed_x1, grad_zero_indices, initials=None, choice=0, Max_loop=5):
    
    vertices1=vertices1_temp=np.hstack((np.asarray(mesh1.vs),np.ones((len(mesh1.vs),1))))
    vertices2=vertices2_temp=np.asarray(mesh2_list[0].vs)
    for i in range(1,len(mesh2_list)):
        vertices1=np.hstack((vertices1, vertices1_temp))
        vertices2=np.hstack((vertices2, np.asarray(mesh2_list[i].vs)))
    print( vertices1.shape )
    print( vertices2.shape )

    H=len(endmembers)


    ## scale the vertices.
    scale=find_scale(vertices1)/2
    vertices1/=scale
    vertices2/=scale



    if initials is not None: #### assume it is numpy array
        x0=initials.copy()
    else:
        x0=np.ones(len(vertices1)*H)/H


    Smooth_Matrix1=create_laplacian(mesh1, H)
    
    Smooth_Matrix=Smooth_Matrix1.T.dot(Smooth_Matrix1) ## bi-laplacian.

    x=x0.ravel().copy()
    y=endmembers.ravel().copy()
    y_old=endmembers.copy().ravel()
    

    fixed_points=np.dot(fixed_x0.reshape((-1, H)), endmembers.reshape((H, -1))).reshape(-1)

    for i in range(Max_loop):
        
        print ("####loop: ", i)
        ### fixed y, solve x
        print ("####solve mixing weights")
        x=optimize(x, vertices1, vertices2, Smooth_Matrix, weights, y, fixed_x0, fixed_x1, grad_zero_indices, choice=choice)
        
        ### fixed x, solve y
        print ("####solve endmembers")
        y=optimize_endmembers(y, vertices1, vertices2, x, fixed_points, choice=choice)

        diff=abs(y-y_old).reshape((H, -1)).sum(axis=-1)
        print ("####endmember changes: ", diff)
        y_old=y.copy()
    
    return x, y.reshape(endmembers.shape)

