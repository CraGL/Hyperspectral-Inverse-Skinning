from __future__ import print_function, division

# python3 is needed.
import scipy.optimize, numpy
import scipy.sparse.linalg

USE_OUR_GRADIENTS = True
if USE_OUR_GRADIENTS:
	print( "Using our gradient functions, not automatic finite differencing ones." )
DEBUG = False

def points_to_data( pts ):
	## pts should be a sequence of n-dimensional points.
	pts = numpy.asarray( pts )
	n = pts.shape[1]
	
	assert len( pts.shape ) == 2
	
	## Make an empty array where the points are the columns with a 1 appended.
	data = numpy.ones( ( n+1, len( pts ) ) )
	data[:-1] = pts.T
	
	return n, data

def repeated_block_diag_times_matrix( block, matrix ):
	# return scipy.sparse.block_diag( [ block ]*( matrix.shape[0]//block.shape[1] ) ).dot( matrix )
	# print( abs( scipy.sparse.block_diag( [ block ]*( matrix.shape[0]//block.shape[1] ) ).dot( matrix ) - numpy.dot( block, matrix.reshape( block.shape[1], -1, order='F' ) ).reshape( -1, matrix.shape[1], order='F' ) ).max() )
	return numpy.dot( block, matrix.reshape( block.shape[1], -1, order='F' ) ).reshape( -1, matrix.shape[1], order='F' )

#### find points that has each channel's max and min value. totally 2*L points (may duplicate)
def min_max( data ):
	### data shape is N*L, L is dimensions. N is data point number
	indices = []
	L = data.shape[1]
	for i in range( L ):
		min_ind = numpy.argmin( data[:,i] )
		max_ind = numpy.argmax( data[:,i] )
		indices.append( min_ind )
		indices.append( max_ind )
	
#	indices, unique_indices = numpy.unique( indices, True )
	endmembers = numpy.asarray(list(set([tuple(data[i]) for i in indices])))

	return endmembers

def cull_interior_points( pts ):
	'''
	Given:
		pts: A sequence of n-dimensional points (e.g. points are rows)
	Returns:
		A subset of the points with some points inside the convex hull removed.
	'''
	dim = pts.shape[1]
	bounded = min_max( pts )
	import scipy.cluster.vq
	clusters = scipy.cluster.vq.kmeans( bounded, dim+1 )
#	import pdb; pdb.set_trace()

	import scipy.spatial
	D = scipy.spatial.Delaunay( clusters[0] )

#	D = scipy.spatial.Delaunay( bounded[:dim+1] )
	bary = D.find_simplex( pts )
	
	return numpy.concatenate( (pts[ ( bary < 0 ) ], bounded), axis=0 )

def make_positive_semidefinite( M ):
	'''
	Given:
		M: A square matrix as a numpy.array.
	Returns:
		P: M modified to be positive semi-definite.
	'''
	
	## Clip the negative eigenvalues.
	
	## Fancier: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
	## See also: https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
	
	## We assume the matrix is symmetric. If not, add it to its transpose and divide by 2.
	# assert abs( M - M.T ).max() < 1e-10
	
	eigenvalues, Q = numpy.linalg.eigh( M )
	
	# assert abs( Q.dot( numpy.diag(eigenvalues).dot( Q.T ) ) - M ).max() < 1e-10
	# assert abs( Q.dot( numpy.diag(eigenvalues).dot( Q.T ) ) - M ).max() < 1e-10
	# assert abs( Q.dot( eigenvalues.reshape(-1,1) * Q.T ) - M ).max() < 1e-10
	
	P = numpy.dot( Q, numpy.maximum( eigenvalues, 0.0 ).reshape(-1,1) * Q.T )
	return P

def to_spmatrix( M ):
	M = scipy.sparse.coo_matrix( M )
	import cvxopt
	return cvxopt.spmatrix( M.data, numpy.asarray( M.row, dtype = int ), numpy.asarray( M.col, dtype = int ) )

def MVES( pts, initial_guess_vertices = None, method = None, linear_solver = None, max_iter = None, min_weight = None ):
	'''
	Given:
		pts: A sequence of n-dimensional points (e.g. points are rows)
		initial_guess_vertices (optional): A sequence of n+1
			n-dimensional points to use as an initial guess for the solution.
	Returns:
		The inverse of n+1 n+1-dimensional points. The inverse of the returned
		matrix has the points as the columns. The last coordinate of each point will
		always be 1.
	'''
	if method is None:
		## This should be 'qp-major' since it works better than 'lp'.
		method = 'lp'	
	
	if min_weight is None:
		min_weight = 0.0
	
	if linear_solver is None:
		linear_solver = 'glpk'
	
	pts = numpy.asfarray( pts )
	print( "All pts #: ", len(pts) )
	
	n, data = points_to_data( pts )
	
	## Our solution is n+1 points, each of which is an n-dimensional points with a 1 appended.
	shape = ( n+1, n+1 )
	
	## From a flat array of unknowns to the convenient matrix shape.
	def unpack( x ):
		V = x.reshape( n+1, n+1 )
		return V
	
	
	## Define the objective function
	def f_volume( x ):
		Vinv = unpack( x )
		
		# invvol = abs( numpy.linalg.det( Vinv ) )
		## Since we are maximizing, we can ignore the absolute value.
		## That's equivalent to swapping columns.
		invvol = numpy.linalg.det( Vinv )
		
		# if invvol > 0:	return invvol
		return -invvol
	
	def f_volume_grad( x ):
		Vinv = unpack( x )
		
		invvol = numpy.linalg.det( Vinv )
		## From: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html#deriv_linear
		return ( -invvol*numpy.linalg.inv(Vinv).T ).ravel()
	
	## It's less work to compute the function and gradient at the same time.
	## Both need the determinant, which is as expensive as matrix inversion.
	def f_volume_with_grad( x ):
		Vinv = unpack( x )
		
		# invvol = abs( numpy.linalg.det( Vinv ) )
		## Since we are maximizing, we can ignore the absolute value.
		## That's equivalent to swapping columns.
		invvol = numpy.linalg.det( Vinv )
		
		## From: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html#deriv_linear
		grad = ( (-invvol)*numpy.linalg.inv(Vinv).T ).ravel()
		
		return -invvol, grad
	
	if DEBUG:
		## Check the gradient.
		print( "Checking the volume gradient." )
		err = scipy.optimize.check_grad( f_volume, f_volume_grad, numpy.random.random( (n+1,n+1) ).ravel() )
		print( 'f_volume gradient is right if this number is ~0:', err )
		err = scipy.optimize.check_grad( lambda x: f_volume_with_grad(x)[0], lambda x: f_volume_with_grad(x)[1], numpy.random.random( (n+1,n+1) ).ravel() )
		print( 'f_volume gradient is right if this number is ~0:', err )
	
	## UPDATE: We now want the correctly-signed log determinant for numerical reasons.
	
	## Define the objective function
	def f_log_volume( x ):
		Vinv = unpack( x )
		
		sign, logdet = numpy.linalg.slogdet( Vinv )
		
		## We want to maximize this quantity, so return it negated.
		# return -sign*logdet
		return -logdet
	
	def f_log_volume_grad( x ):
		Vinv = unpack( x )
		
		## For the log determinant, the derivative is simpler:
		## https://math.stackexchange.com/questions/1233187/compute-the-derivative-of-the-log-of-the-determinant-of-a-with-respect-to-a
		return ( -numpy.linalg.inv(Vinv).T ).ravel()
	
	def f_log_volume_hess( x ):
		Vinv = unpack( x )
		
		## For the log determinant, the derivative is simpler:
		## https://math.stackexchange.com/questions/1233187/compute-the-derivative-of-the-log-of-the-determinant-of-a-with-respect-to-a
		# return ( -numpy.linalg.inv(Vinv).T ).ravel()
		## And then the second derivative:
		## http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html#deriv_det
		Vinvinv = numpy.linalg.inv(Vinv)
		result = ( numpy.kron( Vinvinv.T, Vinvinv ) )
		## WHY OH WHY DO WE HAVE TO DO THIS CRAZY RESHAPE AND TRANSPOSE THING?
		## UPDATE: I believe it's because we do row-major vectorization when
		##         the identities we looked up are for column-major vectorization.
		bigdim = (n+1)*(n+1)
		return result.reshape( bigdim, n+1, n+1 ).transpose((0,2,1)).reshape( bigdim, bigdim )
	
	def f_log_volume_hess_inv( x ):
		Vinv = unpack( x )
		result = ( numpy.kron( Vinv.T, Vinv ) )
		## WHY OH WHY DO WE HAVE TO DO THIS CRAZY RESHAPE AND TRANSPOSE THING?
		## UPDATE: I believe it's because we do row-major vectorization when
		##         the identities we looked up are for column-major vectorization.
		bigdim = (n+1)*(n+1)
		return result.T.reshape( bigdim, n+1, n+1 ).transpose((0,2,1)).reshape( bigdim, bigdim ).T
	
	def test_hessian():
		# n = 3
		numpy.random.seed(0)
		R = numpy.random.random( ( n+1, n+1 ) ).ravel()
		
		hess = f_log_volume_hess( R )
		print( "Hess inverse? (should show identity)" )
		hessinv = f_log_volume_hess_inv(R)
		print( numpy.abs( hess.dot( hessinv ) - numpy.identity(hess.shape[0]) ).sum() )
		print( "Hess symmetric?", numpy.abs( hess - hess.T ).sum() )
		print( "Hess OK?" )
		import hessian
		# Hfd = hessian.hessian( R, grad = f_log_volume_grad )
		Hfd = hessian.hessian( R, f = f_log_volume, epsilon = 1e-5 )
		print( numpy.average( numpy.abs( ( Hfd - hess ) ) ) )
	
	if DEBUG:
		## Check the gradient.
		print( "Checking the log volume gradient." )
		R = (numpy.random.random( (n+1,n+1) )*2).ravel()
		err = scipy.optimize.check_grad( f_log_volume, f_log_volume_grad, R )
		print( 'f_log_volume gradient is right if this number is ~0:', err )
		# print( 'f_volume:', f_volume( R ) )
		# print( 'f_log_volume:', f_log_volume( R ) )
		# print( 'det:', numpy.linalg.det( R.reshape(n+1,n+1) ) )
		# print( 'slogdet:', numpy.linalg.slogdet( R.reshape(n+1,n+1) ) )
		# print( 'slogdet:', numpy.linalg.slogdet( R.reshape(n+1,n+1) )[0]*numpy.exp(numpy.linalg.slogdet( R.reshape(n+1,n+1) )[1]) )
		# print( scipy.optimize.approx_fprime( R, f_log_volume, numpy.sqrt(numpy.finfo(float).eps) ) )
		# print( f_log_volume_grad( R ) )
		
		test_hessian()
	
	
	## Set up the constraints.
	constraints = []
	
	## Constrain the barycentric coordinates to be positive.
	def g_bary( x ):
		Vinv = unpack( x )
		
		# print( Vinv )
		bary = numpy.dot( Vinv, data )
		return bary.ravel()
	def g_bary_jac( x ):
		## From: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html#deriv_linear 
		# return scipy.sparse.kron( scipy.sparse.identity(n+1), data.T )
		return scipy.sparse.block_diag( [ data.T ]*(n+1) )
	def g_bary_jac_cvxopt_spmatrix( x ):
		return to_spmatrix( g_bary_jac( x ) )
	def g_bary_jac_dense( x ):
		return numpy.kron( numpy.identity(n+1), data.T )
	def g_bary_rhs():
		b = numpy.zeros( (n+1)*data.shape[1] )
		b[:] = min_weight
		return b
	
	if DEBUG:
		print( "Checking the positive barycentric coordinate constraint gradient." )
		for i in range(n+1):
			err = scipy.optimize.check_grad( lambda x: g_bary(x)[i], lambda x: g_bary_jac_dense(x)[i], numpy.random.random( (n+1,n+1) ).ravel() )
			print( 'g_bary gradient is right if this number is ~0:', err )
	
	## Constrain the bottom row of the inverse (aka the homogeneous coordinates) to be all ones.
	def g_ones( x ):
		Vinv = unpack( x )
		dp = numpy.dot( numpy.ones(n+1), Vinv )
		dp[-1] -= 1.
		return dp
	def g_ones_jac( x ):
		## From: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html#deriv_linear
		return scipy.sparse.kron( numpy.ones((1,n+1)), scipy.sparse.identity(n+1) )
	def g_ones_jac_cvxopt_spmatrix( x ):
		return to_spmatrix( g_ones_jac( x ) )
	def g_ones_jac_dense( x ):
		return numpy.kron( numpy.ones((1,n+1)), numpy.identity(n+1) )
	def g_ones_rhs():
		b = numpy.zeros( n+1 )
		b[-1] = 1
		return b
	def g_ones_rhs_alt():
		## From: Robust Minimum Volume Simplex Analysis for Hyperspectral Unmixing (A. Agathos, J. Li, J. M. Bioucas-Dias, A. Plaza 2014 European Signal Processing Conference (EUSIPCO))
		# rhs = numpy.dot( numpy.dot( numpy.ones( (1,data.shape[1]) ), data.T ), numpy.linalg.inv( numpy.dot( data, data.T ) ) )
		rhs = numpy.linalg.solve( numpy.dot( data, data.T ), numpy.dot( numpy.ones( (1,data.shape[1]) ), data.T ).T ).T
		## It's always the same as our right-hand-side: [ 0, 0, ..., 0, 0, 1 ]
		print( "g_ones_rhs_alt:", rhs )
		return rhs.ravel()
	
	if DEBUG:
		print( "Checking the homogeneous coordinate constraint gradient." )
		for i in range(n+1):
			err = scipy.optimize.check_grad( lambda x: g_ones(x)[i], lambda x: g_ones_jac_dense(x)[i], numpy.random.random( (n+1,n+1) ).ravel() )
			print( 'g_ones gradient is right if this number is ~0:', err )
	
	def valid_initial( pts ):
		## pts has each n-dimensional point as a row.
		origin = pts.min(axis=0)
		## Offset the points so that they are bounded on all-but-one side by the
		## axis-aligned planes through the origin.
		adj_pts = pts - origin
		## Find the offset of the diagonal plane (x1+x2+x3+x4 ... - D = 0)
		## that puts all offset points under the plane.
		D = adj_pts.sum( axis=1 ).max()
		## Now make the n+1 initial guess points with each point as a row.
		x0 = numpy.zeros( (n+1, n+1) )

#		numpy.fill_diagonal( x0[:n, :n], adj_sum[ ind ]+0.1 )
#		x0[-1, :-1].fill( -0.1 )
#		x0[:,:-1] = x0[:,:-1] - translation
#		x0[:,-1] = numpy.ones( n+1 )


		## We have an extra column that should be all 1's for the homogeneous coordinate.
		x0[:,-1] = 1.
		## The simplex is the origin and all points (c,0,0,...), (0,c,0,0,...), (0,0,c,0,0,...), ...
		## The first n points are D along the coordinate axes.
		x0[:n, :n] = numpy.eye( n )*D
		## The last point is the origin (the array was initialized to zeros).
		## Offset all points so that 0 is the origin.
		x0[:,:-1] += origin
		
		## Verify that all points are inside.
		bary = numpy.linalg.inv( x0.T ).dot( data )
		eps = 1e-8
		assert ( bary >= -eps ).all()
		assert ( bary <= 1+eps ).all()
		assert ( abs( bary.sum(0) - 1.0 ) < eps ).all()
		
		x0 = numpy.linalg.inv( x0.T ).ravel()
		print( "inital volume:", f_volume( x0 ) )
		print( "inital log volume:", f_log_volume( x0 ) )
		return x0
	
	## Make an initial guess.
	if initial_guess_vertices is None:
		x0 = valid_initial( pts )
#		x0 = numpy.identity(n+1).ravel()
		# x0 = numpy.random.random( (n,n+1) ).ravel()
	else:
		x0 = numpy.ones( (n+1, n+1) )
		x0[:,:-1] = initial_guess_vertices
		x0[:,:-1] = x0[:,:-1] + numpy.random.random( (n+1,n) )*1
		x0 = numpy.linalg.inv( x0.T ).ravel()
		
	def callback_volume(xk):
		Vinv = numpy.linalg.inv( unpack( xk ) )
		print( "current volume: ", numpy.linalg.det( Vinv ) )
		
	iteration = [0]
	def show_progress( x ):
		iteration[0] += 1
		print("Iteration", iteration[0])
	
	MAX_ITER = max_iter
	if MAX_ITER is None:
		MAX_ITER = 1000
	
	iter_num = 0
	
	## Solve.
	solution = numpy.linalg.inv( unpack( x0 ) )
	if method == "IPOPT" or method == "ipopt":	
		print( "Solve MEVS with pyipopt." )		
		import pyipopt
		pyipopt.set_loglevel( 2 ) # 1: moderate log level of PyIPOPT				
		nvar = (n+1)*(n+1)
		x_L = numpy.ones( nvar )*pyipopt.NLP_LOWER_BOUND_INF
		x_U = numpy.ones( nvar )*pyipopt.NLP_UPPER_BOUND_INF
		nieq = (n+1)*pts.shape[0]
		neq = n+1
		ncon = neq + nieq
		g_L = numpy.zeros( ncon )
		g_U = numpy.ones( ncon )
		g_U[-neq:].fill(0.)
		
		from scipy.sparse import csr_matrix
		sp_g_ones_jac = csr_matrix(g_ones_jac(x0))
		nnzj = ncon*nvar #nieq*(n+1) + sp_g_ones_jac.getnnz()
		nnzh = 0 #int((n+1)*(n+2)/2)
		eval_f = f_log_volume
		eval_grad_f = f_log_volume_grad
		def eval_g(x):
			return numpy.concatenate((g_bary(x)[:nieq], g_ones(x)))
		def eval_jac_g(x, flag):
			if flag: 
				pattern = numpy.array([range(nvar)])
				return ( numpy.repeat(pattern,ncon,axis=1).ravel(), numpy.repeat(pattern,ncon,axis=0).ravel() )
			else:
				return numpy.concatenate((g_bary_jac(x)[:nieq],g_ones_jac(x)), axis=0).ravel()
				
		nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
		nlp.int_option('max_iter', MAX_ITER)
		
		x, zl, zu, constraint_multipliers, obj, status = nlp.solve( x0 )
		solution = x
	elif method == "IP" or method == "lp":
		print( "Solve MEVS with cvxopt ip solver." )	
		## Linear test:
		import cvxopt
		x0 = x0[:,numpy.newaxis]
		all_x = [ (f_log_volume( numpy.array(x0) ), x0) ]
		
		## set invariant parameters
		G = -g_bary_jac( x0 )
		h = g_bary_rhs()
		A = g_ones_jac( x0 )
		b = g_ones_rhs()
		sparse_G = to_spmatrix(G)
		sparse_A = to_spmatrix(A)
		try:
			while True:		
				c = f_log_volume_grad( x0 )
				solution = cvxopt.solvers.lp( cvxopt.matrix(c), sparse_G, cvxopt.matrix(h), sparse_A, cvxopt.matrix(b), solver=linear_solver )
				# Hc = numpy.linalg.solve( make_positive_semidefinite( f_log_volume_hess( x0 ) ), f_log_volume_grad( x0 ) )
				# solution = cvxopt.solvers.lp( cvxopt.matrix(Hc), sparse_G, cvxopt.matrix(h), sparse_A, cvxopt.matrix(b), solver=linear_solver )
				# Hc = numpy.dot( f_log_volume_hess_inv( x0 ), f_log_volume_grad( x0 ) )
				# solution = cvxopt.solvers.lp( cvxopt.matrix(Hc*0.9+c*0.1), sparse_G, cvxopt.matrix(h), sparse_A, cvxopt.matrix(b), solver=linear_solver )
	
				x = numpy.array( solution['x'] )
				fx = f_log_volume( x )
				print( "Current log volume: ", fx  )
				all_x.append( ( fx, x ) )
				iter_num += 1
				if( numpy.allclose( x, x0, rtol=1e-02, atol=1e-05 ) ):
					print("all close!")
					break
				## More generally, check if we are in a cycle:
				# if [ x - 
				elif( iter_num>MAX_ITER/2 and abs( fx - f_log_volume(x0) ) <= 0.1 ):
					print("log volume is close!")
					break
				elif iter_num >= MAX_ITER:
					print("Exceed the maximum number of iterations!")
					break
	
				x0 += 0.95*(x - x0)
		
		except KeyboardInterrupt:
			print( "Terminated by KeyboardInterrupt." )
		
		print( "# LP Iteration: ", iter_num )
		if iter_num >= MAX_ITER:
			curr_volume = f_log_volume( x0 )
			for i, item in enumerate(all_x):
				if item[0] < curr_volume:
					curr_volume = item[0]
					x0 = item[1]
# 		import pdb; pdb.set_trace()	
		print( "Final log volume:", f_log_volume( numpy.array(x0) ) )
		solution = numpy.linalg.inv( unpack( numpy.array(x0) ) )
		
	elif method == "QP" or method == "qp" or method == "qp-major":
		print( "Solve MEVS with cvxopt qp solver." )	
		import cvxopt
		x0 = x0[:,numpy.newaxis]
		all_x = [ (f_log_volume( numpy.array(x0) ), x0) ]
		
		## set invariant parameters
		#G = -g_bary_jac( x0 )
		#sparse_G = to_spmatrix(G)
		sparse_G = -g_bary_jac_cvxopt_spmatrix( x0 )
		h = g_bary_rhs()
		
		# A = g_ones_jac( x0 )
		# sparse_A = to_spmatrix(A)
		sparse_A = g_ones_jac_cvxopt_spmatrix( x0 )
		b = g_ones_rhs()
		
		try:
			while True:
				## update solver parameters.
				P = f_log_volume_hess( x0 )
				q = f_log_volume_grad( x0 )
				if method == 'qp-major':
					# Following: Robust Minimum Volume Simplex Analysis for Hyperspectral Unmixing (A. Agathos, J. Li, J. M. Bioucas-Dias, A. Plaza 2014 European Signal Processing Conference (EUSIPCO))
					q = q-P.dot( x0 ).squeeze()
					
					## Turn P into a diagonal matrix.
					P = numpy.diag( numpy.diag( P ) )
				else:
					P = make_positive_semidefinite( P )
				## solve
				## MOSEK is so much faster and better!
				solution = cvxopt.solvers.qp( cvxopt.matrix(P), cvxopt.matrix(q), sparse_G, cvxopt.matrix(h), sparse_A, cvxopt.matrix(b), solver = 'mosek' )
				# solution = cvxopt.solvers.qp( cvxopt.matrix(P), cvxopt.matrix(q), sparse_G, cvxopt.matrix(h), sparse_A, cvxopt.matrix(b) )
				if solution['status'] == 'optimal':
					x = numpy.array( solution['x'] )
				else:
					print( "Solution status not optimal:", solution['status'] )
					## Copy x0 over, and the optimization will terminate due to no change.
					x = x0.copy()
				fx = f_log_volume( x )
				print( "Current log volume: ", fx )
				
				iter_num += 1
				
				## Line search as mentioned in the Agathos paper.
				## Was the last solution better?
				while all_x[-1][0] < fx:
					## Bisect the distance between the last solution and the current one.
					x = 0.5*( x0 + x )
					fx = f_log_volume( x )
					
					print( "Volume increased! Bisecting. New log volume:", fx )
					
					## Break this line search if we are too close.
					if numpy.allclose( x, x0 ):
						break
				
				## Have we converged?
				if( numpy.allclose( x, x0, rtol=1e-02, atol=1e-05 ) ):
					print("all close!")
					break
				elif iter_num >= MAX_ITER:
					print("Exceed the maximum number of iterations!")
					break
				all_x.append( ( fx, x ) )
				x0 = x
		except KeyboardInterrupt:
			print( "Terminated by KeyboardInterrupt." )
		
		print( "# QP Iteration: ", iter_num )
		if iter_num >= MAX_ITER:
			curr_volume = f_log_volume( x0 )
			for i, item in enumerate(all_x):
				if item[0] < curr_volume:
					curr_volume = item[0]
					x0 = item[1]
		print( "Final log volume:", f_log_volume( numpy.array(x0) ) )
		solution = numpy.linalg.inv( unpack( numpy.array(x0) ) )
	
	elif method.lower() == "sisal":
		print( "Solve MEVS with SISAL." )
		x0 = x0[:,numpy.newaxis]
		f0 = f_log_volume( x0 )
		all_x = [ (f0, x0) ]
		
		Lambda = 1.
		Tao = (n+1)*1000/len(pts)
		Mu = 1e-6
		def hnorm( X ): return numpy.maximum( -X, 0.0 ).sum()
		def soft( X, beta ): return numpy.maximum( numpy.abs( X + beta/2.0 ) - beta/2.0, 0.0 )*numpy.sign(X)
		def allclose( x1, x2 ): numpy.allclose( x1, x2, rtol=1e-03, atol=1e-06 )
		
		# A = g_bary_jac( x0 )
		Ablock = data.T
		Areps = n+1
		Ashape = ( Ablock.shape[0]*Areps, Ablock.shape[1]*Areps )
		## A should be #pts*dimension-by-dimension^2
		# assert A.shape[0] > A.shape[1]
		assert Ablock.shape[0] > Ablock.shape[1]
		
		# A.T * A
		AtAblock = Ablock.T.dot(Ablock)
		
		# F = Mu * numpy.eye(AtA.shape[0]) + Tao*AtA
		F = Mu * numpy.eye(AtAblock.shape[0]) + Tao*AtAblock
		Finv = numpy.linalg.inv(F)
		B = g_ones_jac(x0)
		## B.dot anything can be sped up as a vertical block sum.
		BFinv = B.dot( scipy.sparse.block_diag( [ Finv ]*Areps ) )
		eq10monster = BFinv.T.dot( scipy.sparse.linalg.inv( B.dot( BFinv.T ) ) ).todense()
		
		a = g_ones_rhs().reshape(-1,1)
		
		l0 = f0 + Lambda*hnorm( repeated_block_diag_times_matrix( Ablock, x0 ) )
		
		try:
			while True:
				g = f_log_volume_grad( x0 ).reshape(-1,1)
				
				## Algorithm 3
				d = numpy.zeros( ( Ashape[0], 1 ) )
				z = numpy.zeros( d.shape )
				alg_3_iterations = 0
				x3 = x0
				while True:
					alg_3_iterations += 1
					b = Mu*x3 - g + Tao*numpy.asfarray( repeated_block_diag_times_matrix( Ablock.T, z + d ) )
					x = numpy.asfarray( repeated_block_diag_times_matrix( Finv, b ) ) - numpy.asfarray( eq10monster.dot( BFinv.dot( b ) - a ) )
					Ax = numpy.asfarray( repeated_block_diag_times_matrix( Ablock, x ) )
					z = soft( Ax - d, Mu/Tao )
					d = d - ( Ax - z )
					# print( x )
					if numpy.allclose( x, x3 ):
						print( "Terminating Algorithm 3 after", alg_3_iterations, "iterations." )
						break
					x3 = x
					if alg_3_iterations >= 1000000:
						print( "Terminating Algorithm 3 after", alg_3_iterations, "iterations (too many)." )
						break
				
				iter_num += 1
				
				fx = f_log_volume( x )
				print( "Current log volume: ", fx )
				lx = fx + Lambda*hnorm( repeated_block_diag_times_matrix( Ablock, x ) )
				
				## Line search.
				## Was the last solution better?
				while l0 < lx:
					## Bisect the distance between the last solution and the current one.
					x = 0.5*( x0 + x )
					fx = f_log_volume( x )
					lx = fx + Lambda*hnorm( repeated_block_diag_times_matrix( Ablock, x ) )
					
					print( "Volume increased! Bisecting. New log volume:", fx )
					
					## Break this line search if we are too close.
					if numpy.allclose( x, x0 ):
						break
				
				## Have we converged?
				if( numpy.allclose( x, x0, rtol=1e-02, atol=1e-05 ) ):
					print("all close!")
					break
				elif iter_num >= MAX_ITER:
					print("Exceed the maximum number of iterations!")
					break
				all_x.append( ( fx, x ) )
				x0 = x
				l0 = lx
		except KeyboardInterrupt:
			print( "Terminated by KeyboardInterrupt." )
		
		print( "SISAL Iterations:", iter_num )
		print( "Final log volume:", f_log_volume( numpy.array(x0) ) )
		solution = numpy.linalg.inv( unpack( numpy.array(x0) ) )
	
	elif method == "BINARY" or method == "binary":
		print( "Solve MEVS with linear binary search." )	
		## Binary search
		G = -g_bary_jac( x0 )
		h = g_bary_rhs()
		A = g_ones_jac( x0 ).todense()
		b = g_ones_rhs()
		from binary_search_opt import min_quad_with_linear_constraints, binary_search
		while True:
			print("Binary outer iteration")
			print( "f(x):", f_log_volume(x0) )
			direction = min_quad_with_linear_constraints( f_log_volume_hess(x0), f_log_volume_grad(x0), A, b )
			x = binary_search( x0, direction, 1.0, G, h, epsilon = 1e-10 )
			if numpy.allclose( x, x0, rtol=1e-03, atol=1e-06 ):
				print("all close!")
				break
			x0 = x
		print( "Final x:", x0 )
		print( "Final x inverse:" )
		print( numpy.linalg.inv( x0.reshape( n+1, n+1 ) ) )
		solution = numpy.linalg.inv( unpack( x0 ) )
	elif method == "SCIPY" or method == "scipy":
		print( "Solve MEVS with scipy.optimize.minimize." )			
		if USE_OUR_GRADIENTS:
			constraints.append( { 'type': 'ineq', 'fun': g_bary, 'jac': g_bary_jac_dense } )
			constraints.append( { 'type': 'eq', 'fun': g_ones, 'jac': g_ones_jac_dense } )
		else:
			constraints.append( { 'type': 'ineq', 'fun': g_bary } )
			constraints.append( { 'type': 'eq', 'fun': g_ones } )
		if USE_OUR_GRADIENTS:
			## Volume:
			# solution = scipy.optimize.minimize( f_volume_with_grad, x0, jac = True, constraints = constraints )
			## Log volume:
			solution = scipy.optimize.minimize( f_log_volume, x0, jac = f_log_volume_grad, hess = f_log_volume_hess, constraints = constraints, callback = show_progress, options = { 'maxiter': MAX_ITER, 'disp': True, 'iprint': 2 } )
		else:
			## Volume:
			# solution = scipy.optimize.minimize( f_volume, x0, constraints = constraints )
			## Log volume:
			solution = scipy.optimize.minimize( f_log_volume, x0, constraints = constraints, callback = show_progress, options = { 'maxiter': MAX_ITER, 'disp': True, 'iprint': 2 } )
		solution = numpy.linalg.inv( unpack( solution.x ) )
	else:
		raise RuntimeError( "Unknown MVES method" )
	
	## Return the solution in a better format.
	
	barycentric = numpy.dot( numpy.linalg.inv( solution ), numpy.concatenate( ( pts.T, numpy.ones((1,pts.shape[0])) ), axis=0 ) )
	if numpy.allclose( barycentric.min(1), numpy.zeros(barycentric.shape[0]) ):
		print( "weight test succeeds." )
	else:
		print( "weight test fails." )
	
	return solution, barycentric.T, iter_num

def MVES_solution_weights_for_points( solution, pts ):
	'''
	Given:
		solution: The solution matrix returns from MVES()
		pts: A sequence of points in the same subspace as MVES was computed
	Returns:
		A sequence of barycentric weights for each point in points as a numpy.array.
	'''
	
	pts = numpy.asarray( pts )
	return numpy.dot( numpy.linalg.inv( solution ), numpy.concatenate( ( pts.T, numpy.ones((1,pts.shape[0])) ), axis=0 ) ).T

def test( method ):
	if method is None: method = 'qp-major'
	
	pts = [ [ 0,1 ], [ 1,0 ], [ -2,0 ], [ 0, 0 ] ]
	# pts = [ [ 0,.9 ], [.1,.9], [ 1,0 ], [ 0, 0 ] ]
	print( 'pts:', pts )
	#numpy.random.seed(0)
	#pts = numpy.random.random_sample((200, 16))
	solution, weights, iterations = MVES( pts, method = method )
	print( 'solution' )
	print( solution )

	print( 'solution * data', '(', len(pts), 'data points)' )
	n, data = points_to_data( pts )
	print( numpy.dot( numpy.linalg.inv( solution ), data ) )

	# simplex = numpy.linalg.inv( solution.x )
	simplex = solution
	print( 'solution simplex', simplex.shape[1], 'points' )
	print( simplex.round(2) )

if __name__ == '__main__':
	import sys
	argv = sys.argv[1:]
	
	test( argv[0] if len( argv ) > 0 else None )
	sys.exit(0)
	
	import scipy.io 
	import format_loader
#	X = scipy.io.loadmat(argv[0])['X'].T
	X = format_loader.load_DMAT(argv[0]).T
	print( 'X.shape:', X.shape )
	T_mat = format_loader.load_Tmat(argv[1]).T
	print( 'T_mat.shape:', T_mat.shape )
	print( 'T_mat' )
	print(T_mat)
	
	from simplex_hull import uncorrellated_space
	project, unproject, scale = uncorrellated_space( X )
	
	# solution = MVES( project( X ), project( T_mat ) )
	solution = MVES( project( X ) )
	print( 'solution' )
	print( solution )
	
	print( 'solution.T (rows are points)' )
	# print( unproject( solution.x[:-1].T ) - T_mat )
	print( unproject( solution[:-1].T ).round(3) )
	# print( 'solution.T 0-th point compared to ground truth 1-st point:' )
	## For the example above, these match:
	# print( unproject( solution.x[:-1].T )[0] - T_mat[1] )
