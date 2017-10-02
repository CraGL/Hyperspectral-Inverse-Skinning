from __future__ import print_function, division
from recordclass import recordclass

from numpy import *

def compute_gamma( weights ):
    '''
    Given:
        weights: A sequence of N-dimensional vectors, each of which represents weights (sums to 1 and has values > 0 and < 1).
    Returns gamma from "Identifiability of the Simplex Volume Minimization Criterion for Blind Hyperspectral Unmixing: The No Pure-Pixel Case" (Chia-Hsiang Lin, Wing-Kin Ma, Wei-Chiang Li, Chong-Yung Chi, ArulMurugan Ambikapathi 2015 arXiv 1406.5273)
    '''
    
    weights = asfarray( weights )
    
    ## check dimension independence
    import simplex_hull
    mapper = simplex_hull.SpaceMapper.Uncorrellated_Space( weights, enable_scale=False )
    
    weights = mapper.project( weights )
    
    ## Get the dimension of the weight vector
    N = weights.shape[1]
    
    ## This is the point we compare to, I think. It's not clearly stated in the paper,
    ## but the figures show it to be this.
    ## It might also be whatever makes the inscribed sphere largest, which
    ## would be a totally different algorithm.
    center_point = 1./N * ones( N )
    
    ## Initialize rho to an impossibly large number.
    rho = amax( [ linalg.norm( row ) for row in W.T ] )
    print( "Purity rho: ", rho )
    
    gamma = 2.
    import scipy.spatial
    hull = scipy.spatial.ConvexHull( weights )
    center_point = hull.points.mean(axis=0)
    print( "center point: ", center_point )
    radius = center_point
    for hyperplane in hull.equations:
    	## http://www.qhull.org/html/index.htm; w*x + b < 0
    	w = hyperplane[:-1]
    	b = hyperplane[-1]
    	assert( dot( w, center_point ) + b < 0 )
    	## https://math.stackexchange.com/questions/1210545/distance-from-a-point-to-a-hyperplane
    	distance_to_hyperplane = abs( dot( w, center_point ) + b ) / linalg.norm( w )
    	# print( "distance: ", distance_to_hyperplane )
    	if distance_to_hyperplane < gamma:
    		gamma = distance_to_hyperplane
    		radius = w / linalg.norm( w ) * distance_to_hyperplane
    
    unproj_radius = mapper.unproject( radius[newaxis, :] )
    gamma = linalg.norm( unproj_radius )
    
    return gamma

def gamma_treshold( N ):
    '''
    If gamma is below the threshold for a simplex of dimension N,
    then "Identifiability of the Simplex Volume Minimization Criterion for Blind Hyperspectral Unmixing: The No Pure-Pixel Case" (Chia-Hsiang Lin, Wing-Kin Ma, Wei-Chiang Li, Chong-Yung Chi, ArulMurugan Ambikapathi 2015 arXiv 1406.5273)
    claims the minimum volume enclosing simplex no longer recovers ground truth.
    '''
    return 1.0/sqrt(N)

## test
if __name__ == '__main__':
	import sys,os
	if len( sys.argv ) != 2:
		print( 'Usage:', sys.argv[0], 'path/to/input.DMAT', file = sys.stderr )
		sys.exit(-1)
		
	path = sys.argv[1]
	import DMAT2MATLAB
	W = DMAT2MATLAB.load_DMAT( path )
	
	gamma = compute_gamma( W.T )
	print( "gamma: ", gamma )
	thres = gamma_treshold( W.shape[0]-1 )
	print( "gamma threshold: ", thres )