from numpy import *

def compute_gamma( weights ):
    '''
    Given:
        weights: A sequence of N-dimensional vectors, each of which represents weights (sums to 1 and has values > 0 and < 1).
    Returns gamma from "Identifiability of the Simplex Volume Minimization Criterion for Blind Hyperspectral Unmixing: The No Pure-Pixel Case" (Chia-Hsiang Lin, Wing-Kin Ma, Wei-Chiang Li, Chong-Yung Chi, ArulMurugan Ambikapathi 2015 arXiv 1406.5273)
    '''
    
    weights = asfarray( weights )
    ## Get the dimension of the weight vector
    N = weights.shape[1]
    
    ## This is the point we compare to, I think. It's not clearly stated in the paper,
    ## but the figures show it to be this.
    ## It might also be whatever makes the inscribed sphere largest, which
    ## would be a totally different algorithm.
    center_point = 1./N * ones( N )
    
    ## Initialize rho to an impossibly large number.
    gamma = 1
    for hyperplane in ConvexHull( weights ):
        gamma = min( rho, distance of hyperplane to `center_point` )
    
    return gamma

def gamma_treshold( N ):
    '''
    If gamma is below the threshold for a simplex of dimension N,
    then "Identifiability of the Simplex Volume Minimization Criterion for Blind Hyperspectral Unmixing: The No Pure-Pixel Case" (Chia-Hsiang Lin, Wing-Kin Ma, Wei-Chiang Li, Chong-Yung Chi, ArulMurugan Ambikapathi 2015 arXiv 1406.5273)
    claims the minimum volume enclosing simplex no longer recovers ground truth.
    '''
    return 1.0/sqrt(N-1)
