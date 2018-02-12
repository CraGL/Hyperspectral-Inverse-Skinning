import autograd.numpy as np
np.set_printoptions( linewidth = 2000 )

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product, Sphere
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient, ParticleSwarm, NelderMead
from pymanopt.solvers import nelder_mead

import flat_metrics

import argparse
## UPDATE: type=bool does not do what we think it does. bool("False") == True.
##		   For more, see https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
str2bool_choices = {'true': True, 'yes': True, 'false': False, 'no': False}
def str2bool(s): return str2bool_choices[s.lower()]
parser = argparse.ArgumentParser( description='karcher and optimization.' )
parser.add_argument('--poses', '-P', type=int, help='Number of poses.')
parser.add_argument('--dim', type=int, help='Ambient dimension.')
parser.add_argument('--ortho', type=int, help='Given flats\' orthogonal dimesion.')
parser.add_argument('--handles', '-H', type=int, help = 'Number of handles.')
parser.add_argument('--mean', type=str, default = 'karcher', choices = ['karcher', 'projection'], help = 'Type of mean.')
parser.add_argument('--test-data', type=str, default = 'random', choices = ['random', 'zero', 'line', 'cube'], help = 'What test data to generate. zero means all flats pass through the origin. lines means there is a line passing through all flats. cube means the edges of a hypercube are specified as lines.')
parser.add_argument('--optimize-from', type=str, choices = [ "random", "centroid" ], help ='What optimization to run (if specified). Choices are "random" and "centroid".')
parser.add_argument('--optimize-solver', type=str, default = "trust", choices = [ "trust", "steepest", "conjugate", "nelder", "particle" ], help ='What optimization solver to use (default "trust" region).')
parser.add_argument('--manifold', type=str, default = 'pB', choices = [ 'pB', 'ssB', 'graff' ], help = 'The manifold to optimize over. Choices are pB (point and nullspace), graff (affine nullspace), ssB (scalar, sphere, and nullspace).')
parser.add_argument('--save', type=str, help = 'If specified, saves p and B to file name.')
parser.add_argument('--load', type=str, help = 'If specified, loads p and B from the file name as the initial guess for optimization.')
parser.add_argument('--lines', type=str2bool, default = False, help = 'Shorthand for --dim 3 --ortho 2 --handles 1.')
parser.add_argument('--centroid-best-p', type=str2bool, default = False, help ='Whether or not to improve the centroid guess with the optimal p (default: False, because it seems to harm optimization).')
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

if args.manifold == 'pB':
    manifold = Product( [ Euclidean(dim), Grassmann(dim, handles) ] )
    def pB_from_X( X ):
        p, B = X
        return p,B
    def X_from_pB( p, B ):
        return [ p, B ]
elif args.manifold == 'ssB':
    manifold = Product( [ Euclidean(1), Sphere(dim), Grassmann(dim, handles) ] )
    def pB_from_X( X ):
        s, S, B = X
        p = s*S
        return p, B
    def X_from_pB( p, B ):
        return [ np.linalg.norm(p), p/np.linalg.norm(p), B ]
elif args.manifold == 'graff':
    manifold = Grassmann(dim+1, handles+1)
    def pB_from_X( X ):
        if abs(X[:-1]).min() < 1e-10:
            ## I expect that in this case, one of the columns is: [ 0 ... 0 1 ]
            ## and the others are [ ... 0 ].
            ## This corresponds to a flat through the origin, and we can simply ignore
            ## the bottom row.
            import pdb
            pdb.set_trace()
            B = X[:-1]
        else:
            B = X[:-1]/X[-1:]
        p = B[:,:1]
        B = B[:,1:] - p
        # This B won't be orthonormal, unlike the other methods. Orthonormalize.
        B = flat_metrics.orthonormalize( B )
        p = p.squeeze()
        return p, B
    def X_from_pB( p, B ):
        X = np.append( np.hstack([ p.reshape(-1,1), p.reshape(-1,1) + B ]), np.ones((1,B.shape[1]+1)), axis = 0 )
        X = flat_metrics.orthonormalize( X )
        return X
else:
    raise RuntimeError( "Unknown problem manifold: %s" % args.manifold )

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
print( "improve the centroid guess with the optimal p:", args.centroid_best_p )
print( "load optimization initial guess from a file:", args.load )
print( "test data:", args.test_data )
print( "mean:", args.mean )
print( "manifold:", args.manifold )
print( "optimization cost function:", "simple" )
print( "manifold:", "E^%s x Grassmann( %s, %s )" % ( dim, dim, handles ) )
print( "====================================================" )

# (2) Define the cost function (here using autograd.numpy)
def cost(X):
    if args.manifold in ('pB','ssB'):
        p,B = pB_from_X( X )
    elif args.manifold == 'graff':
        ## graff_div False works so much better!
        graff_div = False
        weight_affine = 1e3
        if graff_div:
            B = X[:-1]/X[-1:]
            Qaffine = np.ones( ( B.shape[1], B.shape[1] ) )
            RHSaffine = np.ones( B.shape[1] )
        else:
            B = X[:-1]
            Qaffine = np.outer( X[-1], X[-1] )
            RHSaffine = X[-1]
    else: raise RuntimeError
    
    sum = 0.
    
    for A,a in flats:
        # a = np.zeros(a.shape)
        AB = np.dot( A, B )
        
        if args.manifold in ('pB','ssB'):
            z = np.dot( np.linalg.inv( np.dot( AB.T, AB ) ), -np.dot( AB.T, np.dot( A, p - a ) ) )
            diff = np.dot( A, p + np.dot( B, z ) - a )
        elif args.manifold == 'graff':
            ## Impose the z sum-to-one constraint via a large penalty.
            z = np.dot(
                    np.linalg.inv( np.dot( AB.T, AB ) + weight_affine * Qaffine ),
                    np.dot( AB.T, np.dot( A, a ) ) + weight_affine * RHSaffine
                    )
            diff = np.dot( A, np.dot( B, z ) - a )
        else: raise RuntimeError
        
        e = np.dot( diff, diff )
        # e = np.sqrt(e)
        sum += e
    
    ## A regularizer to make p as small as possible.
    ## UPDATE: This doesn't speed up convergence at all!
    # sum += 1e-5 * np.dot( p,p )
    
    return sum

problem = Problem(manifold=manifold, cost=cost)

## Compute the handle-dimensional centroid of the orthogonal space,
## the point on the handle-dimensional manifold whose principal angles to all
## the orthogonal spaces is smallest.
## That should be directions along which the flats are most separated.
## A flat parallel to those dimensions can reduce the distance a lot.

## The geodesic centroid. Is a good starting guess.
if args.mean == 'karcher':
    from pymanopt_karcher_mean import compute_centroid as compute_mean
## Projection distance doesn't work very well.
elif args.mean == 'projection':
    from pymanopt_karcher_mean import compute_projection_mean as compute_mean


## I think this packing isn't what we want for graff manifold.
# centroid = compute_mean( manifold, [ X_from_pB( a, A.T ) for A, a in flats ] )
if args.manifold == 'pB':
    centroid = compute_mean( manifold, [ ( a, A.T ) for A, a in flats ] )
elif args.manifold == 'ssB':
    centroid = compute_mean( manifold, [ ( np.linalg.norm(a), a/np.linalg.norm(a), A.T ) for A, a in flats ] )
elif args.manifold == 'graff':
    ## https://en.wikipedia.org/wiki/Affine_Grassmannian_(manifold)
    ## The orthogonal space next to -rhs
    ## UPDATE: Zero here is averaging just linear subspaces.
    ##         It gets lower error on random tests than the right thing (-A*a) or the wrong thing (A*a).
    ##         It also gets lower error than pB centroid.
    ##         Perhaps because we are on a higher dimensional manifold, so we get to use translation?
    ##         But it makes the trust region solver struggle a lot.
    ## UPDATE 2: I believe the centroid will keep the zeros in the last coordinate intact.
    ##           That's degenerate, because we intersect the columns with last coordinate = 0 <=> divide by it.
    ## TODO Q: Do we preserve orthogonality if we take some directions and put a 1 as the last column?
    ## A: No. Consider two 1D spaces (so orthonormalization is just normalization), v1 and v2.
    ##    Since they are orthogonal, v1.v2 = 0. Then extending them with a 1 and normalizing gives us:
    ##    [v1 1].[v2 1]/(|[v1 1]|*|[v2 1]|) = ( 0 + 1 )/( sqrt(1 + 1)*sqrt(1 + 1) ) = 1/2.
    centroid = compute_mean( manifold, [ flat_metrics.orthonormalize( np.hstack( [ A, -0*np.dot( A, a ).reshape(-1,1) ] ).T ) for A, a in flats ] )
    ## Let's put an epsilon as the last coordinate to keep the optimization from exploding.
    print( "last coordinates:", centroid[-1] )
    centroid[-1] = 1e-4
else: raise RuntimeError

Xopt = centroid

print( "Final cost:", cost( Xopt ) )

# Is zero in the solution flat?
p,B = pB_from_X( Xopt )
print( 'p.T:' )
print( p.T )
print( 'B.T:' )
print( B.T )
p_closest_to_origin = flat_metrics.canonical_point( p, B )
dist_to_origin = np.linalg.norm( p_closest_to_origin )
print( "Distance to the flat from the origin:", dist_to_origin )
p_best = flat_metrics.optimal_p_given_B_for_flats_ortho( B, flats )
print( '|p - p_best|:', np.linalg.norm( p_closest_to_origin - p_best ) )
print( 'p_best.T:', p_best.T )
if args.centroid_best_p:
    ## Pass this p along (and pack it back into Xopt)
    print( "Adopting the best p" )
    p = p_best
    Xopt = X_from_pB( p, B )

if args.save is not None:
    np.savez_compressed( args.save, p = p_closest_to_origin, B = B )
    print( "Saved:", args.save )

if args.optimize_from is not None or args.load is not None:
    if args.optimize_solver == 'trust':
        solver = TrustRegions()
    elif args.optimize_solver == 'conjugate':
        solver = ConjugateGradient()
    elif args.optimize_solver == 'steepest':
        solver = SteepestDescent()
    elif args.optimize_solver == 'particle':
        solver = ParticleSwarm()
    elif args.optimize_solver == 'nelder':
        solver = NelderMead()
    else: raise RuntimeError
    
    if args.load is not None:
        print( "Loading initial guess from a file:", args.load )
        loaded = np.load( args.load )
        p = loaded['p']
        B = loaded['B']
        Xopt = X_from_pB( p, B )
        
        print( "Optimizing the initial guess with the simple original cost function." )
        Xopt2 = solver.solve(problem, x=Xopt)
        
    elif args.optimize_from == 'centroid':
        print( "Optimizing the Karcher mean with the simple original cost function." )
        Xopt2 = solver.solve(problem, x=Xopt)
    elif args.optimize_from == 'random':
        print( "Optimizing from random with the simple original cost function." )
        Xopt2 = solver.solve(problem)
    else:
        raise RuntimeError( "Unknown --optimize-from parameter: %s" % args.optimize_from )
    print( "Final cost:", cost( Xopt2 ) )
    
    p2,B2 = pB_from_X( Xopt2 )
    print( 'p2.T:' )
    print( p2.T )
    print( 'B2.T:' )
    print( B2.T )
    p2_closest_to_origin = flat_metrics.canonical_point( p2, B2 )
    dist2_to_origin = np.linalg.norm( p2_closest_to_origin )
    print( "Distance to the flat from the origin:", dist2_to_origin )
    # p2_best = flat_metrics.canonical_point( flat_metrics.optimal_p_given_B_for_flats_ortho( B2, flats ), B2 )
    ## optimal_p_given_B_for_flats_ortho() should already return the smallest norm p.
    p2_best = flat_metrics.optimal_p_given_B_for_flats_ortho( B2, flats )
    print( '|p2 - p2_best|:', np.linalg.norm( p2_closest_to_origin - p2_best ) )
    print( 'p2_best.T:' )
    print( p2_best.T )
    
    if args.save is not None:
        np.savez_compressed( args.save, p = p2_closest_to_origin, B = B2 )
        print( "Saved:", args.save )
