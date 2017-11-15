# -*- coding: utf-8 -*-
from __future__ import print_function, division

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
	### vertices1 is N*P*4 (homogeneous), vertices2 is N*P*3. X0 is N*(P*3*4), P is pose number 
	N=len(vertices1)
	Matrix=x0.reshape((N, -1, 3, 4))

	reconstruct_vertices2=np.multiply(Matrix, vertices1.reshape((N, -1, 1, 4))).sum(axis=-1)

	return (reconstruct_vertices2-vertices2.reshape((N,-1,3))).reshape(-1)



def objective_func(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level):
	
	M=int(len(x0)/len(vertices1))
	P=int(M/12)
	
	obj=objective_func_vector(x0, vertices1, vertices2)
	val=np.square(obj).sum()/P
	
	
	W_svd=0.0
	W_rotation=0.0
	W_rotation1=0.0
	W_rotation2=0.0
	W_translation=0.0
	W_spatial=0.0
	
	
	if 'W_svd' in weights:
		W_svd=weights["W_svd"]
	if 'W_rotation' in weights:
		W_rotation=weights["W_rotation"]
	if 'W_rotation1' in weights:
		W_rotation1=weights["W_rotation1"]
	if 'W_rotation2' in weights:
		W_rotation2=weights["W_rotation2"]
	if 'W_translation' in weights:
		W_translation=weights["W_translation"]
	if 'W_spatial' in weights:
		W_spatial=weights["W_spatial"]
		
	
	if W_svd!=0.0:
		Matrix=x0.reshape((-1,M))
		L=Matrix-Matrix.mean(axis=0).reshape((1,-1))
		s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
		

		val+=((-1.0)/(1+s[-control_level:]**2)).sum()*len(vertices1)*W_svd/control_level

		# val+=((-1.0)/(1+s[-control_level:]**2)).max()*len(vertices1)*W_svd
		
		# val+=s[-control_level:].sum()*len(vertices1)*W_svd/control_level
		
		# val+=((-1.0)/(1+10*s[-control_level:]**2)).sum()*len(vertices1)*W_svd/control_level


		
		
	if W_rotation!=0.0:
		
		temp1=x0.reshape((-1,3,4))[:,:,:3]
		temp2=temp1.transpose((0,2,1))
		identities=np.repeat(np.identity(3).reshape((1,-1)), len(temp1), 0).ravel()
		val+=np.square((temp1[:,:,:,np.newaxis]*temp2[:,np.newaxis,:,:]).sum(axis=-2).ravel()-identities).sum()*W_rotation/(9*P)
	 
	

	if W_rotation1!=0.0 or W_rotation2!=0.0:
		
		temp1=x0.reshape((-1,3,4))[:,:,:3]
		temp2=temp1.transpose((0,2,1))
		RTR=(temp1[:,:,:,np.newaxis]*temp2[:,np.newaxis,:,:]).sum(axis=-2).ravel()
		
		RTRsqure=np.square(RTR)
		inds=np.repeat(np.array([[0,4,8]]), len(temp1), 0)+ np.arange(len(temp1)).reshape((-1,1))*9	 #### diganl element index in ravel matrix.
		
		
		diagnal_term=np.square(RTR[inds.ravel()]-np.ones(len(inds.ravel()))).sum()
		
		non_diagnal_sum_square=RTRsqure.sum()-RTRsqure[inds.ravel()].sum()
		
		val+=diagnal_term*W_rotation1/(3*P)
		val+=non_diagnal_sum_square*W_rotation2/(6*P)
	
	
	
	if W_translation!=0.0:
		
		temp1=x0.reshape((-1,3,4))[:,:,3]
		val+=np.square(temp1).sum()*W_translation/(3*P)
		
		


	if W_spatial!=0.0:
		#### this is ok, but not supported by autograd library to compute gradient.
		val+=np.dot(x0,Smooth_Matrix.dot(x0))*W_spatial/M
	
	return val



def gradient_objective_func(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level):
	
	W_spatial=weights['W_spatial']
	weights['W_spatial']=0.0 #### turn off, because it seems autograd cannot support scipy.sparse matrix's dot product.
	grad=elementwise_grad(objective_func,0)
	g=grad(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level)
	weights['W_spatial']=W_spatial ### recover
	M=vertices1.shape[1]*vertices2.shape[1]

	if W_spatial!=0.0:
		g2=2*Smooth_Matrix.dot(x0)*W_spatial/M
		g+=g2

	return g




def objective_func_and_gradient(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level):
	obj=objective_func(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level)
	grad=gradient_objective_func(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level)
	return obj, grad
	
	


def optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level):


	start = time.clock()

	res = scipy.optimize.minimize(objective_func, x0, args=(vertices1, vertices2, Smooth_Matrix, weights, control_level)
			,jac = gradient_objective_func
			# ,options={'gtol':1e-4, 'ftol': 1e-4}
			,method='L-BFGS-B'
		 )

	x=res["x"]
	print( res["success"] )
	end = time.clock()

	# print 'took ', (end-start), ' seconds.'

	return x



def optimize_basinhoppings(x0, vertices1, vertices2, Smooth_Matrix, weights, control_level):

	start = time.clock()

	minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, 
						"args": (vertices1, vertices2, Smooth_Matrix, weights, control_level) }
	
	res = scipy.optimize.basinhopping(objective_func_and_gradient, x0, minimizer_kwargs=minimizer_kwargs, 
									  niter=20, 
									  stepsize=0.1,
									  T=2.0
									 )

	x=res.x
	end = time.clock()

	# print 'took ', (end-start), ' seconds.'

	return x


def run_one(mesh1, mesh2_list, outprefix, weights, initials=None, option=3):
	vertices1=vertices1_temp=np.hstack((np.asarray(mesh1.vs),np.ones((len(mesh1.vs),1))))
	vertices2=vertices2_temp=np.asarray(mesh2_list[0].vs)
	for i in range(1,len(mesh2_list)):
		vertices1=np.hstack((vertices1, vertices1_temp))
		vertices2=np.hstack((vertices2, np.asarray(mesh2_list[i].vs)))
	print( vertices1.shape )
	print( vertices2.shape )
	
	M=12*len(mesh2_list)
	
	
	if initials is not None: #### assume it is numpy array
		x0=initials.copy()
	else:
		x0=np.ones(len(vertices1)*M)/M
		
	
	Smooth_Matrix=create_laplacian(mesh1, M)
	
	

	def stop_criteria(rmse_list, rmse, thres=0.001, ratio=3):
		if len(rmse_list)==0:
			return False
		
		if (rmse>thres and rmse>(max(rmse_list)*ratio)) or rmse>0.01:
			return True
	  
		return False
		
		
	if option==4:

		loop=0
		before=0
		after=M-1
	 
		flag=True
		while loop<10:
			print( "############################" )
			print( "loop: ", loop )
			print( weights )
			rmse_list=[]
			after=M-1

			while True:
				i=(before+after)/2
				print( "round: ", i+1 )
				x0_copy=x0.copy()
				### directly on minimize() with L-BFGS-B
				transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

				transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
				L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
				s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
				x0=transformation_matrix.ravel()
				print( "singular values: ", s[:-(i+1)], s[-(i+1):].max(), s[-(i+1):].argmax() )
				rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
				print( "recontruction error: ", rmse )
				if stop_criteria(rmse_list, rmse):
					if flag==True:
						after=i ### record next loop's start index for minimize (start+1)th to 1st smallest singular value
						flag=False
					if loop>0: 
						after=i
						x0=x0_copy.copy()
						break
				else:
					before=i
				rmse_list.append(rmse)
				
				if i==(before+after)/2:
					break
			
  
			# weights["W_svd"]*=0.7
			weights["W_rotation"]*=0.8
			
			# weights["W_rotation1"]+=0.01
			# weights["W_rotation2"]+=0.01
			# weights["W_translation"]+=0.01
			# weights["W_spatial"]+=0.01

			loop+=1

		transformation_matrix=x0
		
	
	if option==3:

		loop=0
		start=0
		flag=True
		while loop<10:
			print( "############################" )
			print( "loop: ", loop )
			print( weights )
			rmse_list=[]
			for i in range(start, M):
				print( "round: ", i+1 )
				x0_copy=x0.copy()
				### directly on minimize() with L-BFGS-B
				transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

				transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
				L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
				s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
				x0=transformation_matrix.ravel()
				print( "singular values: ", s[:-(i+1)], s[-(i+1):].max(), s[-(i+1):].argmax() )
				rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
				print( "recontruction error: ", rmse )
				if stop_criteria(rmse_list, rmse):
					if flag==True:
						start=i-1 ### record next loop's start index for minimize (start+1)th to 1st smallest singular value
						flag=False
					if loop>0: 
						start=i-1
						x0=x0_copy.copy()
						break
				rmse_list.append(rmse)
			
  
			weights["W_svd"]*=0.7
			weights["W_rotation"]*=0.8
			
			# weights["W_rotation1"]+=0.01
			# weights["W_rotation2"]+=0.01
			# weights["W_translation"]+=0.01
			# weights["W_spatial"]+=0.01

			loop+=1

		transformation_matrix=x0

			
	elif option==2: ##### suitable for small w_svd value.
		
		for i in range(M):
			print( "round: ", i )
			### directly on minimize() with L-BFGS-B
			transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

			### basinhopping on L-BFGS-B
			# transformation_matrix=optimize_basinhoppings(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

			transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
			L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
			s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
			x0=transformation_matrix.ravel()
			print( "singular values: ", s.round(3) )
			rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
			print( "recontruction error: ", rmse )

			
	elif option==1:
		loop=0
		while loop<10:
			print( "######loop: ", loop )
			print( weights )
			transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, M)
			
			transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
			L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
			s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
			x0=transformation_matrix.ravel()
			print( "singular values: ", s.round(3) )
			rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
			print( "recontruction error: ", rmse )
			
			weights["W_svd"]*=0.9
			weights["W_rotation"]*=0.9
			loop+=1



	diff=objective_func_vector(transformation_matrix.ravel(), vertices1, vertices2)

	diff=np.square(diff)

	print( 'max diff: ', sqrt(diff).max() )
	print( 'median diff', median(sqrt(diff)) )
	print( 'RMSE: ', sqrt(diff.sum()/(len(x0)/12)) )
	
	return transformation_matrix.ravel()

def run_one_ssd(mesh1, all_poses, outprefix, weights, x0=None, option=3):
	'''
	Input:
		mesh1: trimesh of the rest pose
		all_poses: array of each pose's vs, P-by-N-by-3
		outprefix: prefix for output name
		weights: dictionary of coefficients used for optimization
		x0: initial estimate of vectorized per-vertex transformation, 12NP dimensional vector
		option: which solving strategy to use
	return:
		transformation_matrix: per-vertex transformation
	'''
	vertices1=vertices1_temp=np.hstack((np.asarray(mesh1.vs),np.ones((len(mesh1.vs),1))))
	vertices2=vertices2_temp=np.asarray(all_poses[0])
	for i in range(1,len(all_poses)):
		vertices1=np.hstack((vertices1, vertices1_temp))
		vertices2=np.hstack((vertices2, np.asarray(all_poses[i])))
	print( vertices1.shape )
	print( vertices2.shape )
	
	M=12*len(all_poses)	
	
	if x0 is None: #### assume it is numpy array
		x0=np.ones(len(vertices1)*M)/M	
	
	Smooth_Matrix=create_laplacian(mesh1, M)
	
	

	def stop_criteria(rmse_list, rmse, thres=0.001, ratio=3):
		if len(rmse_list)==0:
			return False
		
		if (rmse>thres and rmse>(max(rmse_list)*ratio)) or rmse>0.01:
			return True
	  
		return False
		
		
	if option==4:

		loop=0
		before=0
		after=M-1
	 
		flag=True
		while loop<10:
			print( "############################" )
			print( "loop: ", loop )
			print( weights )
			rmse_list=[]
			after=M-1

			while True:
				i=(before+after)/2
				print( "round: ", i+1 )
				x0_copy=x0.copy()
				### directly on minimize() with L-BFGS-B
				transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

				transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
				L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
				s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
				x0=transformation_matrix.ravel()
				print( "singular values: ", s[:-(i+1)], s[-(i+1):].max(), s[-(i+1):].argmax() )
				rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
				print( "recontruction error: ", rmse )
				if stop_criteria(rmse_list, rmse):
					if flag==True:
						after=i ### record next loop's start index for minimize (start+1)th to 1st smallest singular value
						flag=False
					if loop>0: 
						after=i
						x0=x0_copy.copy()
						break
				else:
					before=i
				rmse_list.append(rmse)
				
				if i==(before+after)/2:
					break
			
  
			# weights["W_svd"]*=0.7
			weights["W_rotation"]*=0.8
			
			# weights["W_rotation1"]+=0.01
			# weights["W_rotation2"]+=0.01
			# weights["W_translation"]+=0.01
			# weights["W_spatial"]+=0.01

			loop+=1

		transformation_matrix=x0
		
	
	if option==3:

		loop=0
		start=0
		flag=True
		while loop<10:
			print( "############################" )
			print( "loop: ", loop )
			print( weights )
			rmse_list=[]
			for i in range(start, M):
				print( "round: ", i+1 )
				x0_copy=x0.copy()
				### directly on minimize() with L-BFGS-B
				transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

				transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
				L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
				s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
				x0=transformation_matrix.ravel()
				print( "singular values: ", s[:-(i+1)], s[-(i+1):].max(), s[-(i+1):].argmax() )
				rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
				print( "recontruction error: ", rmse )
				if stop_criteria(rmse_list, rmse):
					if flag==True:
						start=i-1 ### record next loop's start index for minimize (start+1)th to 1st smallest singular value
						flag=False
					if loop>0: 
						start=i-1
						x0=x0_copy.copy()
						break
				rmse_list.append(rmse)
			
  
			weights["W_svd"]*=0.7
			weights["W_rotation"]*=0.8
			
			# weights["W_rotation1"]+=0.01
			# weights["W_rotation2"]+=0.01
			# weights["W_translation"]+=0.01
			# weights["W_spatial"]+=0.01

			loop+=1

		transformation_matrix=x0

			
	elif option==2: ##### suitable for small w_svd value.
		
		for i in range(M):
			print( "round: ", i )
			### directly on minimize() with L-BFGS-B
			transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

			### basinhopping on L-BFGS-B
			# transformation_matrix=optimize_basinhoppings(x0, vertices1, vertices2, Smooth_Matrix, weights, i+1)

			transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
			L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
			s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
			x0=transformation_matrix.ravel()
			print( "singular values: ", s.round(3) )
			rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
			print( "recontruction error: ", rmse )

			
	elif option==1:
		loop=0
		while loop<10:
			print( "######loop: ", loop )
			print( weights )
			transformation_matrix=optimize(x0, vertices1, vertices2, Smooth_Matrix, weights, M)
			
			transformation_matrix=transformation_matrix.reshape((len(vertices1),M))
			L=transformation_matrix-transformation_matrix.mean(axis=0).reshape((1,-1))
			s = np.linalg.svd(L, full_matrices=True, compute_uv=False)
			x0=transformation_matrix.ravel()
			print( "singular values: ", s.round(3) )
			rmse=np.sqrt(np.square(objective_func_vector(x0, vertices1, vertices2)).sum()/(len(x0)/12))
			print( "recontruction error: ", rmse )
			
			weights["W_svd"]*=0.9
			weights["W_rotation"]*=0.9
			loop+=1



	diff=objective_func_vector(transformation_matrix.ravel(), vertices1, vertices2)

	diff=np.square(diff)

	print( 'max diff: ', sqrt(diff).max() )
	print( 'median diff', median(sqrt(diff)) )
	print( 'RMSE: ', sqrt(diff.sum()/(len(x0)/12)) )
	
	return transformation_matrix.ravel()







	

	
	