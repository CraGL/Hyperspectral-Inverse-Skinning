import autograd.numpy as np

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient

# (1) Instantiate a manifold
poses = 10
handles = 10
# p, B
manifold = Product( ( Euclidean(12*poses), Grassmann(12*poses, handles) ) )

#method = 'AndersonDuffin'
method = 'block'
#method = 'power'
power = 5
assert method in ('AndersonDuffin', 'block', 'power')

## (1b) Generate data
## TODO: Zero energy test data.
N = 100
Q = 3*poses
if method == 'block':
    Q = 12*poses - Q
np.random.seed(0)
## Create a bunch of orthonormal rows and a point (rhs)
flats = [ ( np.random.random(( Q, 12*poses )), np.random.random(12*poses) ) for i in range(N) ]
## Orthonormalize the rows
flats = [ ( np.linalg.svd( A, full_matrices=False )[2][:Q], a ) for A, a in flats ]
## The block method needs different input.
if method == 'block':
    flats = [ ( A.T, np.dot( A.T, a[:A.shape[0]] ) ) for A, a in flats ]

# (2) Define the cost function (here using autograd.numpy)
def cost(X):
    p,B = X
    sum = 0.
    
    if method in ('AndersonDuffin', 'power'):
        I = np.eye(12*poses)
        P_Bortho = (I - np.dot( B, B.T ) )
    
    for A,a in flats:
        if method == 'AndersonDuffin':
            ## The Anderson-Duffin formula
            ## https://mathoverflow.net/questions/108177/intersection-of-subspaces
            P_Aortho = np.dot( A.T, A )
            ## Is the pseudoinverse necessary?
            if type(B) == np.ndarray:
                mr = np.linalg.matrix_rank( P_Bortho + P_Aortho )
                if mr < P_Bortho.shape[0]:
                    print( "Matrix not full rank! We should be using pseudoinverse. (%s instead of %s)" % ( mr, P_Bortho.shape[0] ) )
                # print( P_Bortho.shape, np.linalg.matrix_rank( P_Bortho + P_Aortho ) )
                # print( np.linalg.svd( P_Bortho + P_Aortho, compute_uv = False ) )
            ## This should be pinv() not inv().
            orthogonal_to_A_and_B = np.dot( 2.*P_Bortho, np.dot( np.linalg.inv( P_Bortho + P_Aortho ), P_Aortho ) )
        elif method == 'block':
            ## Compute the projection onto the intersection of orthogonal spaces.
            ## B is a null space. A is a null space. The desired projection matrix
            ## is I - projection onto union of A and B.
            ATB = np.dot( A.T, B )
            UL = np.linalg.inv( np.eye(Q) - np.dot( ATB, ATB.T ) )
            LR = np.linalg.inv( np.eye(handles) - np.dot( ATB.T, ATB ) )
            UR = np.dot( -ATB, LR )
            LL = UR.T
            
            left = np.dot( A, UL ) + np.dot( B, LL )
            right = np.dot( A, UR ) + np.dot( B, LR )
            
            parallel_to_A_or_B = np.dot( left, A.T ) + np.dot( right, B.T )
            orthogonal_to_A_and_B = np.eye(12*poses) - parallel_to_A_or_B
        elif method == 'power':
            P_Aortho = np.dot( A.T, A )
            orthogonal_to_A_and_B = np.dot( P_Aortho, P_Bortho )
            ## Take the matrix to the power 2^(power-1)
            ## The error decreases exponentially in the power.
            ## See:
            ## Projectors on intersections of subspaces (Adi Ben–Israel 2013 Contemporary Mathematics)
            ## http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf
            for i in range(power-1):
                orthogonal_to_A_and_B = np.dot( orthogonal_to_A_and_B, orthogonal_to_A_and_B )
        
        diff = np.dot( orthogonal_to_A_and_B, p - a )
        e = np.dot( diff, diff )
        sum += e
    return sum

problem = Problem(manifold=manifold, cost=cost)

# (3) Instantiate a Pymanopt solver
# solver = SteepestDescent()
solver = TrustRegions()
# solver = ConjugateGradient()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print(Xopt)
