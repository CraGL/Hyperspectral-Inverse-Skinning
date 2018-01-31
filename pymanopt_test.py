import autograd.numpy as np

from pymanopt.manifolds import Stiefel, Grassmann, Euclidean, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient

# (1) Instantiate a manifold
poses = 10
handles = 10
manifold = Product( ( Euclidean(12*poses), Grassmann(12*poses, handles) ) )

# (2) Define the cost function (here using autograd.numpy)
N = 1000
Q = 3*poses
AndersonDuffin = True
if not AndersonDuffin:
    Q = 12*poses - Q
np.random.seed(0)
## Make orthonormal flats as a nullspace and a point.
flats = [ ( np.random.random(( Q, 12*poses )), np.random.random(Q if AndersonDuffin else 12*poses) ) for i in range(N) ]
flats = [ ( np.linalg.svd( A, full_matrices=False )[2][:Q], a ) for A, a in flats ]
if AndersonDuffin:
    flats = [ ( A, np.dot( A.T, a ) ) for A, a in flats ]
else:
    flats = [ ( A.T, a ) for A, a in flats ]

def cost(X):
    p,B = X
    sum = 0.
    
    if AndersonDuffin:
        I = np.eye(12*poses)
        P_Bortho = (I - np.dot( B, B.T ) )
    
    for A,a in flats:
        if AndersonDuffin:
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
        else:
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
