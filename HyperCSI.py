"""
Python implementation of HyperCSI, http://www.ee.nthu.edu.tw/cychi/source_code_download-e.php

Written by Songrun Liu
"""

from __future__ import print_function, division
from recordclass import recordclass

from numpy import *
import time
import scipy

# function [A_est, S_est, time] = HyperCSI(X,N)
# t0 = clock;

def sort_eig(X):
	eigenValues,eigenVectors = linalg.eig(X)
	idx = eigenValues.argsort() #[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	return eigenValues, eigenVectors

def compute_bi(a0,i,N):
	Hindx = setdiff1d(range(N),[i])
	A_Hindx = a0[:,Hindx]
	A_tilde_i = matrix(A_Hindx[:,:N-2]-repeat(A_Hindx[:,N-2][:,newaxis],N-2,1))
	bi = A_Hindx[:,N-2]-a0[:,i]
	bi = (eye(N-1) - A_tilde_i*(linalg.pinv(A_tilde_i.T*A_tilde_i))*A_tilde_i.T)*bi[:,newaxis]
	bi = bi/linalg.norm(bi)
	return bi

def SPA(Xd,L,N):
	#  Input
	#  Xd is dimension-reduced (DR) data matrix.
	#  L is the number of pixels.   
	#  N is the number of endmembers.
	# ---------------------------------------------------------------------
	#  Output
	#  alpha_tilde is an (N-1)-by-N matrix whose columns are DR purest pixels.

	#----------- Define default parameters------------------
	con_tol = 1e-8; # the convergence tolence in SPA
	num_SPA_itr = N; # number of iterations in post-processing of SPA
	N_max = N; # max number of iterations

	#------------------------ initialization of SPA ------------------------
	assert(len(Xd.shape) == 2)
	assert(Xd.shape[1] == L)
	Xd_t = ones((Xd.shape[0]+1, L))
	Xd_t[:-1, :] = Xd
	
	array_sums = sum( Xd_t**2, axis=0 )
	ind = argmax(array_sums)
	val = array_sums[ind]
	
	A_set = Xd_t[:,ind][:,newaxis]
	index = [ind]
	for i in range(1,N):
		XX = dot((eye(N_max) - dot(A_set, linalg.pinv(A_set))), Xd_t)
		
		array_sums = sum( XX**2, axis=0 )
		ind = argmax(array_sums)
		val = array_sums[ind]
		
		A_set = concatenate((A_set, Xd_t[:,ind][:,newaxis]), axis=1)
		index.append(ind)
	alpha_tilde = Xd[:, index]

	#------------------------ post-processing of SPA ------------------------
	current_vol = linalg.det( alpha_tilde[:,:N-1] - repeat(alpha_tilde[:,N-1][:,newaxis],N-1,1) );
	for jjj in range(num_SPA_itr):
		for i in range(N):
			b = compute_bi(alpha_tilde,i,N)
			b = -b
			idx = argmax(dot(b.T,Xd))
			alpha_tilde[:,i] = Xd[:,idx]
		new_vol = linalg.det( alpha_tilde[:,:N-1] - repeat(alpha_tilde[:,N-1][:,newaxis],N-1,1) )
		if (new_vol - current_vol)/current_vol  < con_tol:
			break
	return alpha_tilde


def hyperCSI(X, N):
	start = time.time()
	#------------------------ Step 1 ------------------------	
	M, L = shape(X)
	d = mean(X, axis=1)
	U = X-repeat(d[:,newaxis],L,1)
	D,eV = sort_eig(matmul(U,U.T))
	
	C = eV[:,M-N+1:]
	Xd = matmul(C.T,U)

	#------------------------ Step 2 ------------------------
	alpha_tilde = SPA(Xd,L,N); # the identified purest pixels

	#------------------------ Step 3 ------------------------
	bi_tilde = compute_bi(alpha_tilde,0,N)
	for i in range(1,N):
		bi_tilde = concatenate((bi_tilde, compute_bi(alpha_tilde,i,N)), axis=1)	# obtain bi_tilde

	r = 0.5*linalg.norm(alpha_tilde[:,0]-alpha_tilde[:,1])
	dist_ai_aj = zeros((N-1, N))
	for i in range(N-1):
		for j in range(i+1,N):
			dist_ai_aj[i,j] = linalg.norm(alpha_tilde[:,i]-alpha_tilde[:,j])
			if 0.5*dist_ai_aj[i,j] < r:
				r = 0.5*dist_ai_aj[i,j]  # compute radius of hyperballs

	Xd_divided_idx = zeros((L,1))
	Xd_divided_idx.fill(-1)
	radius_square = r**2
	for k in range(N):
		IDX_alpha_i_tilde = nonzero( sum( (Xd- repeat(alpha_tilde[:,k][:,newaxis],L,1) )**2,axis=0 )  < radius_square )
		Xd_divided_idx[IDX_alpha_i_tilde] = k 	 # compute the hyperballs

	#------------------------ Step 4 ------------------------
	b_hat = []
	h_hat = zeros((N,1))
	for i in range(N):
		Hi_idx = setdiff1d(range(N),[i])
		pi_k = zeros((Xd.shape[0], N-1))
		for k in range(N-1):
			Ri_k = Xd[:, (Xd_divided_idx == Hi_idx[k]).squeeze() ]	
			idx = argmax(bi_tilde[:,i].T*Ri_k)
			pi_k[:,k] = Ri_k[:,idx]  # find N-1 affinely independent points for each hyperplane
		temp_bi = compute_bi(concatenate((pi_k, alpha_tilde[:,i][:,newaxis]),axis=1),N-1,N)	
		if b_hat == []:		b_hat = temp_bi
		else:				b_hat = concatenate((b_hat,temp_bi),axis=1)
 		h_hat[i,0] = amax(b_hat[:,i].T*Xd)
	
	#------------------------ Step 5 & Step 6 ------------------------
	comm_flag = 1
	# comm_flag = 1 in noisy case: bring hyperplanes closer to the center of data cloud
	# comm_flag = 0 when no noise: Step 5 will not be performed (and hence c = 1)

	eta = 0.9; # 0.9 is empirically good choice for endmembers in USGS library
	alpha_hat = zeros((N-1,N))
	for i in range(N):
		bbbb = b_hat
		ccconst = h_hat
		bbbb = delete(bbbb,i,axis=1)
		ccconst = delete(ccconst,i,axis=0)
		alpha_hat[:,i] = array(linalg.pinv(bbbb.T)*ccconst).squeeze()
	
	if comm_flag == 1:
		VV = dot(C,alpha_hat)
		UU = repeat(d[:,newaxis],N,axis=1)
		closed_form_optval = max( 1 , amax( divide(-VV,UU) ) ) # c.T in Step 5
		c = closed_form_optval/eta
		h_hat = h_hat/c
		alpha_hat = alpha_hat/c
	A_est = dot(C,alpha_hat) + repeat(d[:,newaxis],N,axis=1) # endmemeber estimates
	
	#------------------------ Step 7 ------------------------
	# Step 7 can be removed if the user do not need abundance estimation
	S_est = divide(repeat(h_hat,L,axis=1)- b_hat.T*Xd, repeat( h_hat-sum( multiply(b_hat,alpha_hat).T, axis=1 ),L,axis=1) )
	S_est[ nonzero(S_est<0) ] = 0
	# end
	end = time.time()
	return A_est, S_est, end-start

import sys,os		
if len( sys.argv ) != 3:
    print( 'Usage:', sys.argv[0], 'path/to/input.mat N', file = sys.stderr )
    sys.exit(-1)

# path = "models/cube4/cube.mat"
# N = 4
argv = sys.argv[1:]
		
import scipy.io	
X = scipy.io.loadmat(argv[0])['X']
N = int(argv[1])
A_est, S_est, time_elapsed = hyperCSI(X, N)
print("A_est: ", A_est)
print("S_est: ", S_est)