from __future__ import print_function, division

from numpy import *

def fro( A ):
    '''
    General Frobenius norm.
    
    test:
    M = arange(100).reshape(10,10)
    print( fro( M ) )
    print( linalg.norm( M, 'fro' ) )
    '''
    
    return linalg.norm( A.ravel() )

def KGError( animation, reconstructed ):
    '''
    Error from "Compression of soft-body animation sequences" (Karni and Gotsman 2004 Computers and Graphics).
    
    Dimensions of both inputs: 3-by-num vertices-by-num frames
    '''
    
    assert animation.shape == reconstructed.shape
    
    E = animation.mean( axis = 1 )[:,None,:]
    top = fro( animation - reconstructed )
    bottom = fro( animation - E )
    return 100 * top/bottom

def centered( animation ):
    '''
    Dimension input and output: 3-by-num vertices-by-num frames
    '''
    assert animation.shape[0] == 3
    assert len( animation.shape ) == 3
    
    ## center
    return animation - animation.mean(axis=1)[:,None,:]

def normalizing_scale( animation ):
    '''
    Scale factor to normalize an animation.
    Does not need animation to be centered.
    
    Divide by the bounding sphere radius
    (or 1/2 the bounding box diagonal?)
    for the first frame to get the
    error metric E_RMS from Kavan et al 2010.
    It is unknown what bounding sphere algorithm
    Kavan et al 2010 used.
    Let's use the bounding box diagonal.
    '''
    diag = animation[:,:,0].max(axis=1) - animation[:,:,0].min(axis=1)
    scale = 1/( 0.5 * linalg.norm( diag ) )
    return scale

def KSOError( animation, reconstructed ):
    '''
    Error from "Fast and efficient skinning of animated meshes" (Ladislav Kavan, Peter-Pike Sloan, Carol O’Sullivan 2010 Computer Graphics Forum).
    
    Dimensions of both inputs: 3-by-num vertices-by-num frames
    '''
    
    ## center
    animation = centered( animation )
    reconstructed = centered( reconstructed )
    
    assert animation.shape == reconstructed.shape
    assert animation.shape[0] == 3
    
    _, num_vertices, num_frames = animation.shape
    
    ## Divide this by the bounding sphere radius
    ## (or 1/2 the bounding box diagonal?)
    ## for the first frame to get the
    ## error metric E_RMS from Kavan et al 2010.
    ## It is unknown what bounding sphere algorithm
    ## Kavan et al 2010 used.
    E_RMS_kavan2010 = 1000*linalg.norm( animation.ravel() - reconstructed.ravel() )/sqrt(prod( animation.shape ))
    return E_RMS_kavan2010

# vertices
n = 100
# frames
T = 40

# an animation
random.seed(0)
A = random.random( ( 3, n, T ) )
# print( A[:,:5,0].T )
## TODO Q: Why does only KGError improve when normalizing?
NORMALIZE = False
if NORMALIZE:
    A = centered( A )
    A *= normalizing_scale( A )
print( abs( A.mean(axis=1) ).max() )
# noise
noise = random.random( ( 3, n, T ) )*.1
if NORMALIZE:
    noise = centered( noise )
# reconstructed
R = A + noise

print( 'KGError:', KGError( A, R ) )
print( 'KSOError:', KSOError( A, R ) )
