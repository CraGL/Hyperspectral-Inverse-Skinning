"""
Compute Convex hull from a set of OBJ poses.

Written by Songrun Liu
"""

from __future__ import print_function, division

import numpy

class SpaceMapper( object ):
	def __init__( self ):
		self.Xavg_ = None
		self.U_ = None
		self.s_ = None
		self.V_ = None
		self.stop_s = None
		self.scale = None
	
	def project( self, correllated_poses ):
		scale = self.scale
		stop_s = self.stop_s
		V = self.V_
		Xavg = self.Xavg_
		if scale is not None:
			return numpy.multiply( numpy.dot( correllated_poses - Xavg, V[:stop_s].T ), scale )
		else:
			return numpy.dot( correllated_poses - Xavg, V[:stop_s].T )

	def unproject( self, uncorrellated_poses ):
		scale = self.scale
		stop_s = self.stop_s
		V = self.V_
		Xavg = self.Xavg_
		if scale is not None:
			return numpy.dot( numpy.divide( uncorrellated_poses, scale ), V[:stop_s] ) + Xavg
		else:
			return numpy.dot( uncorrellated_poses, V[:stop_s] ) + Xavg
			 
	@staticmethod
	def PCA_Dimension( X, threshold = 1e-6 ):
		## Subtract the average.
		X = numpy.array( X )
		Xavg = numpy.average( X, axis = 0 )[numpy.newaxis,:]
		Xp = X - Xavg
	
		U, s, V = numpy.linalg.svd( Xp, full_matrices = False, compute_uv = True )
	
		## The first index less than threshold
		stop_s = len(s) - numpy.searchsorted( s[::-1], threshold )
		
		return stop_s
	
# 	PCA_Dimension = staticmethod( PCA_Dimension )
	
	## Formalize the above with functions, from Yotam's experiments
	@staticmethod
	def Uncorrellated_Space( X, enable_scale=True, threshold = None, dimension = None ):
		space_mapper = SpaceMapper()
		
		if threshold is not None and dimension is not None:
			raise RuntimeError( "Please only set one of the optional parameters: threshold or dimension" )
		
		if threshold is None and dimension is None:
			threshold = 1e-6
	
		## Subtract the average.
		Xavg = numpy.average( X, axis = 0 )[numpy.newaxis,:]
		# print("Xavg: ", Xavg)
		Xp = X - Xavg
		space_mapper.Xavg_ = Xavg
	
		U, s, V = numpy.linalg.svd( Xp, full_matrices = False, compute_uv = True )
		space_mapper.U_ = U
		space_mapper.s_ = s
		space_mapper.V_ = V
	
		if threshold is not None:
			## The first index less than threshold
			stop_s = len(s) - numpy.searchsorted( s[::-1], threshold )
		else:
			assert dimension is not None
			stop_s = dimension
		# print( "s: ", s )
		# print( "stop_s: ", stop_s )
		space_mapper.stop_s = stop_s
	
		## Change scale to something that makes the projection of the points
		## have unit size in each dimension...
		if enable_scale:
			scale = numpy.array( [1./(max(x)-min(x)) for x in numpy.dot( Xp, V[:stop_s].T ).T ] )
			# print( "scale: ", scale )
			space_mapper.scale = scale 
	
		return space_mapper
		
# 	Uncorrellated_Space = staticmethod( Uncorrellated_Space )
