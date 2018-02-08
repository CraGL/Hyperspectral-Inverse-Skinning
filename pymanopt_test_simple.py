import autograd.numpy as np
np.set_printoptions( linewidth = 2000 )

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient, ParticleSwarm

# (1) Instantiate a manifold
poses = 10
handles = 5
N = 20
# p, B
manifold = Product( [ Euclidean(12*poses), Grassmann(12*poses, handles) ] )

## (1b) Generate data
## TODO: Zero energy test data.
Q = 3*poses
np.random.seed(0)
## Create a bunch of orthonormal rows and a point (rhs)
flats = [ ( np.random.random(( Q, 12*poses )), np.random.random(12*poses) ) for i in range(N) ]
## Orthonormalize the rows
flats = [ ( np.linalg.svd( A, full_matrices=False )[2][:Q], a ) for A, a in flats ]

test_data = 'line'
if test_data == 'zero':
    ## With a known solution at zero:
    flats = [ ( A, np.zeros(a.shape) ) for A, a in flats ]
elif test_data == 'line':
    ## With a known solution along a line:
    flats = [ ( A, i*np.ones(a.shape) ) for i, ( A, a ) in enumerate( flats ) ]
    print( "The solution should have slope:", 1./np.sqrt(12*poses) )
elif test_data == 'random':
    ## This is the default.
    pass
else:
    raise RuntimeError( "Unknown test data request" )

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

# (3) Instantiate a Pymanopt solver
solver_args = {}
# solver = SteepestDescent()
# solver = ConjugateGradient()
solver = TrustRegions()
## Delta_bar = 100 made a huge difference (running without it printed a suggestion to do it).
# solver_args = { 'Delta_bar': 100. }
# solver = ParticleSwarm()

# let Pymanopt do the rest
Xopt = solver.solve(problem, **solver_args)
# print(Xopt)

print( "Final cost:", cost( Xopt ) )

# Is zero in the solution flat?
p, B = Xopt
print( 'p:' )
print( p )
print( 'B:' )
print( B )
import flat_metrics
p_closest_to_origin = flat_metrics.canonical_point( p, B )
dist_to_origin = np.linalg.norm( p_closest_to_origin )
print( "Distance to the flat from the origin:", dist_to_origin )
