import autograd.numpy as np
np.set_printoptions( linewidth = 2000 )

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient, ParticleSwarm
from pymanopt.solvers import nelder_mead

# (1) Instantiate a manifold
poses = 10
dim = 12*poses
Q = 3*poses
handles = 5

## Lines in 3D
lines = True
if lines:
    dim = 3
    Q = 2
    handles = 1

N = 200
# p, B

START_FROM_CENTROID = True

manifold = Product( [ Euclidean(dim), Grassmann(dim, handles) ] )

## (1b) Generate data
## TODO: Zero energy test data.
np.random.seed(0)
## Create a bunch of orthonormal rows and a point (rhs)
flats = [ ( np.random.random(( Q, dim )), np.random.random(dim) ) for i in range(N) ]
## Orthonormalize the rows
flats = [ ( np.linalg.svd( A, full_matrices=False )[2][:Q], a ) for A, a in flats ]

test_data = 'random'
if test_data == 'zero':
    ## With a known solution at zero:
    flats = [ ( A, np.zeros(a.shape) ) for A, a in flats ]
elif test_data == 'line':
    ## With a known solution along a line:
    flats = [ ( A, i*np.ones(a.shape) ) for i, ( A, a ) in enumerate( flats ) ]
    print( "The solution should have slope:", 1./np.sqrt(dim) )
elif test_data == 'random':
    ## This is the default.
    pass
elif test_data == 'cube':
    assert dim == 3
    assert Q == 2
else:
    raise RuntimeError( "Unknown test data request" )

print( "====================================================" )
print( "ambient dimension:", dim )
print( "number of given flats:", N )
print( "given flat orthogonal dimension:", Q )
print( "affine subspace dimension:", handles )
print( "use optimization to improve the centroid:", START_FROM_CENTROID )
print( "test data:", test_data )
print( "optimization cost function:", "simple" )
print( "manifold:", "E^%s x Grassmann( %s, %s )" % ( dim, dim, handles ) )
print( "====================================================" )

# (2) Define the cost function (here using autograd.numpy)
def cost(X):
    p,B = X
    sum = 0.
    
    for A,a in flats:
        # a = np.zeros(a.shape)
        AB = np.dot( A, B )
        z = np.dot( np.linalg.inv( np.dot( AB.T, AB ) ), -np.dot( AB.T, np.dot( A, p - a ) ) )
        diff = np.dot( A, p + np.dot( B, z ) - a )
        e = np.dot( diff, diff )
        # e = np.sqrt(e)
        sum += e
    
    ## A regularizer to make p as small as possible.
    ## UPDATE: This doesn't speed up convergence at all!
    # sum += 1e-5 * np.dot( p,p )
    
    return sum

problem = Problem(manifold=manifold, cost=cost)

from pymanopt_karcher_mean import compute_centroid
centroid = compute_centroid( manifold, [ ( a, A.T ) for A, a in flats ] )
Xopt = centroid

print( "Final cost:", cost( Xopt ) )

# Is zero in the solution flat?
p, B = Xopt
print( 'p.T:' )
print( p.T )
print( 'B.T:' )
print( B.T )
import flat_metrics
p_closest_to_origin = flat_metrics.canonical_point( p, B )
dist_to_origin = np.linalg.norm( p_closest_to_origin )
print( "Distance to the flat from the origin:", dist_to_origin )

solver = TrustRegions()
if START_FROM_CENTROID:
    print( "Optimizing the Karcher mean with the simple original cost function." )
    Xopt2 = solver.solve(problem, x=Xopt)
else:
    print( "Optimizing from random with the simple original cost function." )
    Xopt2 = solver.solve(problem)
print( "Final cost:", cost( Xopt2 ) )

p2, B2 = Xopt2
print( 'p2.T:' )
print( p.T )
print( 'B2.T:' )
print( B2.T )
p2_closest_to_origin = flat_metrics.canonical_point( p2, B2 )
dist2_to_origin = np.linalg.norm( p2_closest_to_origin )
print( "Distance to the flat from the origin:", dist2_to_origin )
