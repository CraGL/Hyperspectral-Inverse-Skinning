import numpy as np 
import scipy
import scipy.sparse
import cvxopt
from trimesh import TriMesh
from format_loader import *

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

def to_spmatrix( M ):
	M = scipy.sparse.coo_matrix( M )
	return cvxopt.spmatrix( M.data, np.asarray( M.row, dtype = int ), np.asarray( M.col, dtype = int ) )

def col_major(M):
	return M.T.ravel()

def col_major_back(x, row, col):
	return x.reshape(col, row).T


def optimize(vertices1, vertices2, Lap, weights, endmembers, fixed_x0, grad_zero_indices, initials=None):

	cvxopt.solvers.options['show_progress'] = True
	# cvxopt.solvers.options['maxiters'] = 30
	# cvxopt.solvers.options['abstol'] = 1e-4
	# cvxopt.solvers.options['reltol'] = 1e-4
	# cvxopt.solvers.options['feastol'] = 1e-100
    

    # options['show_progress'] True/False (default: True)
    # options['maxiters'] positive integer (default: 100)
    # options['refinement']  positive integer (default: 0)
    # options['abstol'] scalar (default: 1e-7)
    # options['reltol'] scalar (default: 1e-6)
    # options['feastol'] scalar (default: 1e-7).


	### cvxopt qp solver: minimize (1/2) x^TPx + q^T x, subject to Gx<=h and Ax = b
	N=len(vertices1) ## vertices number
	P=vertices1.shape[1]//4 ## pose number
	H=len(endmembers) ### handle number

	print (N, P, H)

	#### build P_matrix and q

	### for data term. |W.dot(E)-W_fixed.dot(E)|^2
	E=endmembers.reshape((H, -1))
	A1=scipy.sparse.kron(E.T, scipy.sparse.identity(N))
	temp=col_major(fixed_x0.reshape((N,H)))
	b1=A1.dot(temp)

	### for spatial term. |Lap.dot(W)-Lap.dot(W_fixed)|^2
	A2=scipy.sparse.kron(scipy.sparse.identity(H), Lap)
	b2=A2.dot(temp)

	W_spatial=0.0
	if 'W_spatial' in weights:
		W_spatial=weights['W_spatial']

	print ('W_spatial', W_spatial)

	### old, when W_spatial is zero, then does not need to consider each term normalization.  
	# P_matrix=2*(A1.T.dot(A1)+W_spatial*A2.T.dot(A2))
	# q=-2*(A1.T.dot(b1)+W_spatial*A2.T.dot(b2))


	### should normalize two term before add them.  
	P_matrix = 2*( A1.T.dot(A1)/P + W_spatial*A2.T.dot(A2) )/H
	q = -2*( A1.T.dot(b1)/P + W_spatial*A2.T.dot(b2)/H )


	### build A,b
	A=scipy.sparse.kron(np.ones((1,H)), scipy.sparse.identity(N)) ### for sum to be 1 per row of mixing weights (N*H)
	b=np.ones(N)

	if len(grad_zero_indices)!=0:
		A_2=np.zeros(N*H)
		A_2[grad_zero_indices]=1.0 ### those zeros values sum should always be zero, during optimization.
		b_2=np.zeros(1)
		A=scipy.sparse.vstack((A,A_2))
		b=np.concatenate((b,b_2))

	### build G,h
	G1=scipy.sparse.identity(N*H)
	G2=-1.0*scipy.sparse.identity(N*H)
	G=scipy.sparse.vstack((G1,G2))
	h1=np.ones(N*H)
	h2=np.zeros(N*H)
	h=np.concatenate((h1,h2))

	# print (initials['x'].shape)

	cvx_initials=None

	# if initials!=None:
	# 	cvx_initials={'x', cvxopt.matrix(initials['x'])}



	solution = cvxopt.solvers.qp( to_spmatrix(P_matrix), cvxopt.matrix(q), 
								  to_spmatrix(G), cvxopt.matrix(h), 
								  to_spmatrix(A), cvxopt.matrix(b),
								  initvals=cvx_initials
								  )

	if solution['status'] == 'optimal':
		print ('optimal')
	else:
		print ("Solution status not optimal:", solution['status'])

	x = np.array( solution['x'] )

	return col_major_back(x, N, H)


def run(mesh1, mesh2_list, weights, endmembers, fixed_x0, grad_zero_indices, initials=None):

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

	Smooth_Matrix=Get_basic_Laplacian_sparse_matrix(mesh1)

	x=optimize(vertices1, vertices2, Smooth_Matrix, weights, endmembers, fixed_x0, grad_zero_indices, initials=initials)

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


def recover_vertices(x0, vertices1, endmembers):
    N=len(vertices1)
    Matrix=np.dot(x0,endmembers)
    Matrix=Matrix.reshape((N, -1, 3, 4))
    reconstruct_vertices2=np.multiply(Matrix, vertices1.reshape((N, -1, 1, 4))).sum(axis=-1)
    return reconstruct_vertices2




