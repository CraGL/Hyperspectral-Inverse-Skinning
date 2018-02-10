import autograd.numpy as np
np.set_printoptions( linewidth = 2000 )

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient, ParticleSwarm
from pymanopt.solvers import nelder_mead

import argparse
parser = argparse.ArgumentParser( description='karcher and optimization.' )
parser.add_argument('--poses', '-P', type=int, help='Number of poses.')
parser.add_argument('--dim', type=int, help='Ambient dimension.')
parser.add_argument('--ortho', type=int, help='Given flats\' orthogonal dimesion.')
parser.add_argument('--handles', '-H', type=int, help = 'Number of handles.')
parser.add_argument('--mean', type=str, default = 'karcher', choices = ['karcher', 'projection'], help = 'Type of mean.')
parser.add_argument('--test-data', type=str, default = 'random', choices = ['random', 'zero', 'line', 'cube'], help = 'What test data to generate. zero means all flats pass through the origin. lines means there is a line passing through all flats. cube means the edges of a hypercube are specified as lines.')
parser.add_argument('--optimize-from', type=str, default = "centroid", choices = [ "random", "centroid" ], help ='What optimization to run (if specified). Choices are "random" and "centroid".')
## UPDATE: type=bool does not do what we think it does. bool("False") == True.
##		   For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
str2bool_choices = {'true': True, 'yes': True, 'false': False, 'no': False}
def str2bool(s): return str2bool_choices[s.lower()]
parser.add_argument('--lines', type=str2bool, default = False, help = 'Shorthand for --dim 3 --ortho 2 --handles 1.')
parser.add_argument('--optimize-p', type=str2bool, default = True, help = 'Whether or not to optimize p.')
args = parser.parse_args()

## Print the arguments.
from pprint import pprint
print( "args:" )
pprint( args )
print( "Re-run with:" )
import sys
try:
    import shlex
    print( ' '.join([ shlex.quote( a ) for a in sys.argv ]) )
except:
    print( ' '.join( sys.argv ) )

# (1) Instantiate a manifold
poses = 10
dim = 12*poses
Q = 3*poses
handles = 5

if args.poses is not None: poses = args.poses
if args.dim is not None: dim = args.dim
if args.ortho is not None: Q = args.ortho
if args.handles is not None: handles = args.handles

## Lines in 3D
if args.lines:
    dim = 3
    Q = 2
    handles = 1

N = 200
# p, B

if args.optimize_p:
    manifold = Product( [ Euclidean(dim), Grassmann(dim, handles) ] )
else:
    manifold = Grassmann(dim, handles)
    p = np.zeros(dim)
    pg = p

## (1b) Generate data
## TODO: Zero energy test data.
np.random.seed(0)
## Create a bunch of orthonormal rows and a point (rhs)
flats = [ ( np.random.random(( Q, dim )), np.random.random(dim) ) for i in range(N) ]
## Orthonormalize the rows
flats = [ ( np.linalg.svd( A, full_matrices=False )[2][:Q], a ) for A, a in flats ]

if args.test_data == 'zero':
    ## With a known solution at zero:
    flats = [ ( A, np.zeros(a.shape) ) for A, a in flats ]
elif args.test_data == 'line':
    ## With a known solution along a line:
    flats = [ ( A, i*np.ones(a.shape) ) for i, ( A, a ) in enumerate( flats ) ]
    print( "The solution should have slope:", 1./np.sqrt(dim) )
elif args.test_data == 'random':
    ## This is the default.
    pass
elif args.test_data == 'cube':
    assert dim == 3
    assert Q == 2
    flats = []
    # flats.append( ( np.array([
    raise NotImplementedError
else:
    raise RuntimeError( "Unknown test data: %s" % args.test_data )

print( "====================================================" )
print( "ambient dimension:", dim )
print( "number of given flats:", N )
print( "given flat orthogonal dimension:", Q )
print( "affine subspace dimension:", handles )
print( "use optimization to improve the centroid:", args.optimize_from )
print( "test data:", args.test_data )
print( "mean:", args.mean )
print( "optimize p:", args.optimize_p )
print( "optimization cost function:", "simple" )
print( "manifold:", "E^%s x Grassmann( %s, %s )" % ( dim, dim, handles ) )
print( "====================================================" )

# (2) Define the cost function (here using autograd.numpy)
def cost(X):
    if args.optimize_p:
        p,B = X
    else:
        B = X
        p = pg
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

if args.mean == 'karcher':
    from pymanopt_karcher_mean import compute_centroid as compute_mean
elif args.mean == 'projection':
    from pymanopt_karcher_mean import compute_projection_mean as compute_mean

if args.optimize_p:
    centroid = compute_mean( manifold, [ ( a, A.T ) for A, a in flats ] )
else:
    centroid = compute_mean( manifold, [ A.T for A, a in flats ] )
Xopt = centroid

print( "Final cost:", cost( Xopt ) )

# Is zero in the solution flat?
if args.optimize_p:
    p, B = Xopt
else:
    B = Xopt
print( 'p.T:' )
print( p.T )
print( 'B.T:' )
print( B.T )
import flat_metrics
p_closest_to_origin = flat_metrics.canonical_point( p, B )
dist_to_origin = np.linalg.norm( p_closest_to_origin )
print( "Distance to the flat from the origin:", dist_to_origin )

if args.optimize_from is not None:
    solver = TrustRegions()
    if args.optimize_from == 'centroid':
        print( "Optimizing the Karcher mean with the simple original cost function." )
        Xopt2 = solver.solve(problem, x=Xopt)
    elif args.optimize_from == 'random':
        print( "Optimizing from random with the simple original cost function." )
        Xopt2 = solver.solve(problem)
    else:
        raise RuntimeError( "Unknown --optimize-from parameter: %s" % args.optimize_from )
    print( "Final cost:", cost( Xopt2 ) )
    
    if args.optimize_p:
        p2, B2 = Xopt2
    else:
        p2 = p
        B2 = Xopt2
    print( 'p2.T:' )
    print( p.T )
    print( 'B2.T:' )
    print( B2.T )
    p2_closest_to_origin = flat_metrics.canonical_point( p2, B2 )
    dist2_to_origin = np.linalg.norm( p2_closest_to_origin )
    print( "Distance to the flat from the origin:", dist2_to_origin )
