from __future__ import print_function, division

import sys
inpath = sys.argv[1]

from numpy import *
import numpy as np
import scipy.io
data = scipy.io.loadmat( inpath )

As = data['A']
as_ortho = data['a_ortho']
as_full = data['a_full']
p = data['p'].ravel()
B = data['B']

if len( sys.argv ) == 3:
    np.savez_compressed( sys.argv[2], p = p, B = B )
    print( "Saved p,B to:", sys.argv[2] )

print( "Number of flats:", len(As) )
print( "Ambient dimension:", As.shape[-1] )
print( "Given flat orthogonal dimension:", As.shape[1] )
print( "Solution flat dimension:", B.shape[1] )

total_a_deviation = 0.
total_dist = 0.

for A, a_ortho, a_full in zip( As, as_ortho, as_full ):
    total_a_deviation += np.linalg.norm( A.dot( a_full ) - a_ortho )
    
    AB = np.dot( A, B )
    lh = np.dot( AB.T, AB )
    rh = -np.dot( AB.T, np.dot(A,p) - a_ortho )
    z = np.linalg.lstsq( lh, rh )[0].ravel()
    dist = np.dot( A, p + B.dot(z) ) - a_ortho
    total_dist += np.dot( dist, dist )

print( "Total a ortho vs full deviation:", total_a_deviation )
print( "Total distance:", total_dist )
