from __future__ import print_function, division

import scipy.optimize, numpy

USE_OUR_GRADIENTS = True
if USE_OUR_GRADIENTS:
	print( "Using our gradient functions, not automatic finite differencing ones." )
DEBUG = True	

def points_to_data( pts ):
	## pts should be a sequence of n-dimensional points.
	pts = numpy.asarray( pts )
	n = pts.shape[1]
	
	assert len( pts.shape ) == 2
	
	## Make an empty array where the points are the columns with a 1 appended.
	data = numpy.ones( ( n+1, len( pts ) ) )
	data[:-1] = pts.T
	
	return n, data

def MVES( pts, initial_guess_vertices = None ):
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
	
	pts = numpy.asfarray( pts )
	
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
		return numpy.kron( numpy.identity(n+1), data.T )
	
	if DEBUG:
		print( "Checking the positive barycentric coordinate constraint gradient." )
		for i in range(n+1):
			err = scipy.optimize.check_grad( lambda x: g_bary(x)[i], lambda x: g_bary_jac(x)[i], numpy.random.random( (n+1,n+1) ).ravel() )
			print( 'g_bary gradient is right if this number is ~0:', err )
	
	if USE_OUR_GRADIENTS:
		constraints.append( { 'type': 'ineq', 'fun': g_bary, 'jac': g_bary_jac } )
	else:
		constraints.append( { 'type': 'ineq', 'fun': g_bary } )
	
	
	## Constrain the bottom row of the inverse (aka the homogeneous coordinates) to be all ones.
	def g_ones( x ):
		Vinv = unpack( x )
		dp = numpy.dot( numpy.ones(n+1), Vinv )
		dp[-1] -= 1.
		return dp
	def g_ones_jac( x ):
		## From: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html#deriv_linear
		return numpy.kron( numpy.ones((1,n+1)), numpy.identity(n+1) )
	
	if DEBUG:
		print( "Checking the homogeneous coordinate constraint gradient." )
		for i in range(n+1):
			err = scipy.optimize.check_grad( lambda x: g_ones(x)[i], lambda x: g_ones_jac(x)[i], numpy.random.random( (n+1,n+1) ).ravel() )
			print( 'g_ones gradient is right if this number is ~0:', err )
	
	if USE_OUR_GRADIENTS:
		constraints.append( { 'type': 'eq', 'fun': g_ones, 'jac': g_ones_jac } )
	else:
		constraints.append( { 'type': 'eq', 'fun': g_ones } )
	
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
		
		print( "inital volume:", numpy.linalg.det( x0 ) )

		return numpy.linalg.inv( x0.T ).ravel()
	
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
	
	iteration = [0]
	def show_progress( x ):
	    iteration[0] += 1
	    print("Iteration", iteration[0])
	
	## Solve.
	if USE_OUR_GRADIENTS:
		## Volume:
		# solution = scipy.optimize.minimize( f_volume_with_grad, x0, jac = True, constraints = constraints )
		## Log volume:
		solution = scipy.optimize.minimize( f_log_volume, x0, jac = f_log_volume_grad, constraints = constraints, callback = show_progress )
	else:
		## Volume:
		# solution = scipy.optimize.minimize( f_volume, x0, constraints = constraints )
		## Log volume:
		solution = scipy.optimize.minimize( f_log_volume, x0, constraints = constraints, callback = show_progress )
	
	## Return the solution in a better format.
	solution.x = numpy.linalg.inv( unpack( solution.x ) )
	# solution.x = unpack( x0 )
	
	barycentric = numpy.dot( numpy.linalg.inv( solution.x ), numpy.concatenate( ( pts.T, numpy.ones((1,pts.shape[0])) ), axis=0 ) )
#	import pdb; pdb.set_trace() 
	if numpy.allclose( barycentric.min(1), numpy.zeros(barycentric.shape[0]) ):
		print( "Initial test succeeds." )
	else:
		print( "Initial test fails." )
	
	return solution

def test():
	pts = [ [ 0,1 ], [ 1,0 ], [ -2,0 ], [ 0, 0 ] ]
	# pts = [ [ 0,.9 ], [.1,.9], [ 1,0 ], [ 0, 0 ] ]
	print( 'pts:', pts )
	#numpy.random.seed(0)
	#pts = numpy.random.random_sample((200, 16))
	solution = MVES( pts )
	print( 'solution' )
	print( solution )

	print( 'solution * data', '(', len(pts), 'data points)' )
	n, data = points_to_data( pts )
	print( numpy.dot( numpy.linalg.inv( solution.x ), data ) )

	# simplex = numpy.linalg.inv( solution.x )
	simplex = solution.x
	print( 'solution simplex', simplex.shape[1], 'points' )
	print( simplex.round(2) )

if __name__ == '__main__':
	import sys
	argv = sys.argv[1:]
	
	test()
	sys.exit(0)
	
	import scipy.io 
	import DMAT2MATLAB
#	X = scipy.io.loadmat(argv[0])['X'].T
	X = DMAT2MATLAB.load_DMAT(argv[0]).T
	print( 'X.shape:', X.shape )
	T_mat = DMAT2MATLAB.load_Tmat(argv[1]).T
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
	print( unproject( solution.x[:-1].T ).round(3) )
	# print( 'solution.T 0-th point compared to ground truth 1-st point:' )
	## For the example above, these match:
	# print( unproject( solution.x[:-1].T )[0] - T_mat[1] )
