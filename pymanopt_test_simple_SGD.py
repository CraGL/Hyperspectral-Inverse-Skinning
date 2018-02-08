import autograd.numpy as np

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient

# (1) Instantiate a manifold
poses = 10
handles = 5
N = 200
# p, B
manifold = Product( [ Euclidean(12*poses), Grassmann(12*poses, handles) ] )

## Stochastic Gradient Descent (SGD)
all_indices = np.arange( N )
BATCH_SIZE = 6

## (1b) Generate data
## TODO: Zero energy test data.
Q = 3*poses
np.random.seed(0)
## Create a bunch of orthonormal rows and a point (rhs)
flats = [ ( np.random.random(( Q, 12*poses )), np.random.random(12*poses) ) for i in range(N) ]
## Orthonormalize the rows
flats = [ ( np.linalg.svd( A, full_matrices=False )[2][:Q], a ) for A, a in flats ]

# (2) Define the cost function (here using autograd.numpy)
def cost(X):
    p,B = X
    sum = 0.
    
    for A,a in flats_batch:
        # a = np.zeros(a.shape)
        AB = np.dot( A, B )
        z = np.dot( np.linalg.inv( np.dot( AB.T, AB ) ), -np.dot( AB.T, np.dot( A, p - a ) ) )
        diff = np.dot( A, p + np.dot( B, z ) - a )
        e = np.dot( diff, diff )
        sum += e
    return sum

problem = Problem(manifold=manifold, cost=cost)

# (3) Instantiate a Pymanopt solver
solver_args = {}
# solver = SteepestDescent( maxiter = 10 )
# solver_args = { 'maxiter': 1 }
# solver = ConjugateGradient( maxiter = 10 )
# solver = ConjugateGradient()
solver = TrustRegions( logverbosity = 2, maxiter = 20 )
## Delta_bar = 100 made a huge difference (running without it printed a suggestion to do it).
# solver_args = { 'Delta_bar': 100. }

# let Pymanopt do the rest
Xopt = None
for iteration in range(1,1001):
    np.random.shuffle( all_indices )
    ## Every 100 iterations try the whole problem.
    if iteration % 5 == 0:
        all_flats_batch = True
        print( "All flats batch!" )
        flats_batch = flats
    else:
        all_flats_batch = False
        flats_batch = [ flats[i] for i in all_indices[:BATCH_SIZE] ]
    
    print( "Starting batch iteration:", iteration )
    solver_args['x'] = Xopt
    Xopt, optlog = solver.solve(problem, **solver_args)
    if all_flats_batch and 'min grad norm reached' in optlog['stoppingreason']:
        print( "Converged!" )
        break
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
