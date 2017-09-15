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
	
	n, data = points_to_data( pts )
	
	## Our solution is n+1 points, each of which is an n-dimensional points with a 1 appended.
	shape = ( n+1, n+1 )
	
	## From a flat array of unknowns to the convenient matrix shape.
	def unpack( x ):
		V = x.reshape( n+1, n+1 )
		return V
	
	
	## Define the objective function
	def f( x ):
		Vinv = unpack( x )
		
		# invvol = abs( numpy.linalg.det( Vinv ) )
		## Since we are maximizing, we can ignore the absolute value.
		## That's equivalent to swapping columns.
		invvol = numpy.linalg.det( Vinv )
		
		return -invvol
	
	def f_grad( x ):
		Vinv = unpack( x )
		invvol = numpy.linalg.det( Vinv )
		## From: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html#deriv_linear
		return ( -invvol*numpy.linalg.inv(Vinv).T ).ravel()
	
	## It's less work to compute the function and gradient at the same time.
	## Both need the determinant, which is as expensive as matrix inversion.
	def f_with_grad( x ):
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
		err = scipy.optimize.check_grad( f, f_grad, numpy.random.random( (n+1,n+1) ).ravel() )
		print( 'f gradient is right if this number is ~0:', err )
		err = scipy.optimize.check_grad( lambda x: f_with_grad(x)[0], lambda x: f_with_grad(x)[1], numpy.random.random( (n+1,n+1) ).ravel() )
		print( 'f gradient is right if this number is ~0:', err )
	
	
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
		for i in range(n+1):
			err = scipy.optimize.check_grad( lambda x: g_ones(x)[i], lambda x: g_ones_jac(x)[i], numpy.random.random( (n+1,n+1) ).ravel() )
			print( 'g_ones gradient is right if this number is ~0:', err )
	
	if USE_OUR_GRADIENTS:
		constraints.append( { 'type': 'eq', 'fun': g_ones, 'jac': g_ones_jac } )
	else:
		constraints.append( { 'type': 'eq', 'fun': g_ones } )
	
	def valid_initial( pts ):
		translation = -pts.min(axis=0)
		adj_pts = pts + translation
		adj_sum = numpy.sum( adj_pts, axis=1 )
		ind = numpy.argmax( adj_sum )
		x0 = numpy.zeros( (n+1, n+1) )
		x0[:n, :n] = numpy.eye( n )*adj_sum[ ind ]
		x0[:,:-1] = x0[:,:-1] - translation
		x0[:,-1] = numpy.ones( n+1 )
		
		print( "inital volumn: ", numpy.linalg.det( x0 ) )
		return numpy.linalg.inv( x0.T ).ravel()
	
	## Make an initial guess.
	if initial_guess_vertices is None:
		x0 = valid_initial( pts )
# 		x0 = numpy.identity(n+1).ravel()
		# x0 = numpy.random.random( (n,n+1) ).ravel()
	else:
		x0 = numpy.ones( (n+1, n+1) )
		x0[:,:-1] = initial_guess_vertices
		x0[:,:-1] = x0[:,:-1] + numpy.random.random( (n+1,n) )*1
		x0 = numpy.linalg.inv( x0.T ).ravel()
# 	import pdb; pdb.set_trace()
	## Solve.
	if USE_OUR_GRADIENTS:
		solution = scipy.optimize.minimize( f_with_grad, x0, jac = True, constraints = constraints )
	else:
		solution = scipy.optimize.minimize( f, x0, constraints = constraints )
	
	## Return the solution in a better format.
	solution.x = numpy.linalg.inv( unpack( solution.x ) )
	# solution.x = unpack( x0 )
	
	return solution

def test():
	# pts = [ [ 0,1 ], [ 1,0 ], [ -2,0 ], [ 0, 0 ] ]
	# pts = [ [ 0,.9 ], [.1,.9], [ 1,0 ], [ 0, 0 ] ]
	numpy.random.seed(0)
	pts = numpy.random.random_sample((20000, 16))
	solution = MVES( pts )
	print( 'solution' )
	print( solution )

	print( 'solution * data', '(', len(pts), 'data points)' )
	n, data = points_to_data( pts )
	print( numpy.dot( solution.x, data ) )

	simplex = numpy.linalg.inv( solution.x )
	print( 'solution simplex', simplex.shape[1], 'points' )
	print( simplex.round(2) )

if __name__ == '__main__':
	import sys
	argv = sys.argv[1:]
	
	import scipy.io 
	import DMAT2MATLAB
# 	X = scipy.io.loadmat(argv[0])['X'].T
	X = DMAT2MATLAB.load_DMAT(argv[0]).T
	print( 'X.shape:', X.shape )
	T_mat = DMAT2MATLAB.load_Tmat(argv[1]).T
	print( 'T_mat.shape:', T_mat.shape )
	print( 'T_mat' )
	print(T_mat)
	
	from convex_hull import uncorrellated_space
	project, unproject = uncorrellated_space( X )
	
	# solution = MVES( project( X ), project( T_mat ) )
	solution = MVES( project( X ) )
	print( 'solution' )
	print( solution )
	
	print( 'solution.T (rows are points)' )
	# print( unproject( solution.x[:-1].T ) - T_mat )
	print( unproject( solution.x[:-1].T ).round(3) )
	print( 'solution.T 0-th point compared to ground truth 1-st point:' )
	## For the example above, these match:
	print( unproject( solution.x[:-1].T )[0] - T_mat[1] )
