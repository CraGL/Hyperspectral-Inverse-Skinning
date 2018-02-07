import autograd.numpy as np

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient

# (1) Instantiate a manifold
poses = 10
handles = 10
N = 100
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
        sum += e
    return sum

problem = Problem(manifold=manifold, cost=cost)

# (3) Instantiate a Pymanopt solver
solver_args = {}
# solver = SteepestDescent()
# solver = ConjugateGradient()
solver = TrustRegions()
## Delta_bar = 100 made a huge difference (running without it printed a suggestion to do it).
solver_args = { 'Delta_bar': 100. }

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
dist_to_origin = np.linalg.norm( np.dot( B.T, 0 - p ) )
print( "Distance to the flat from the origin:", dist_to_origin )
