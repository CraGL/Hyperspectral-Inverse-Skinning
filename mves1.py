from __future__ import print_function, division

import scipy.optimize, numpy

def MVES( pts ):
	## pts should be a sequence of n-dimensional points.
	pts = numpy.asarray( pts )
	n = pts.shape[1]
	
	assert len( pts.shape ) == 2
	
	## Make an empty array where the points are the columns with a 1 appended.
	data = numpy.ones( ( n+1, len( pts ) ) )
	data[:-1] = pts.T
	
	## Our solution is n+1 points, each of which is an n-dimensional points with a 1 appended.
	shape = ( n+1, n+1 )
	
	def unpack( x ):
		V = numpy.ones( shape )
		V[:-1] = x.reshape( n, n+1 )
		return V

	def f( x ):
		V = unpack( x )
		
		vol = abs( numpy.linalg.det( V ) )
		
		return vol
	
	constraints = []
	
	'''
	def g( x ):
		V = unpack( x )
		
		# print( V )
		Vinv = numpy.linalg.inv( V )
		bary = numpy.dot( Vinv, data )
		return abs( bary ).sum()
	
	constraints.append( { 'type': 'ineq', 'fun': g } )
	'''
	
	## Constrain the barycentric coordinates to be positive.
	def gen_g_positive( constraints ):
		for i in range( len( pts ) ):
			
			def gen_g_positive_i( i ):
				def g_positive( x ):
					V = unpack( x )
					
					Vinv = numpy.linalg.inv( V )
					# print( Vinv )
					bary = numpy.dot( Vinv, data[:,i] )
					return bary
				return g_positive
			
			constraints.append( { 'type': 'ineq', 'fun': gen_g_positive_i(i) } )
	
	gen_g_positive( constraints )
	
	x0 = numpy.identity(n+1)[:-1].ravel()
	solution = scipy.optimize.minimize( f, x0, method='SLSQP', constraints = constraints )
	solution.x = unpack( solution.x )
	
	return solution

# pts = [ [ 0,1 ], [ 1,0 ], [ -2,0 ], [ 0,0 ] ]
pts = [ [ 0,.9, 0., 1 ], [ .1, .9, 0.5, -0.5 ], [ 1,0, -.2, 0.8 ], [ 0, 0, -0.7, 0.3 ] ]
pts = numpy.random.random_sample((20, 8))

solution = MVES( pts )
print( solution )
print( solution.x.round(2) )
