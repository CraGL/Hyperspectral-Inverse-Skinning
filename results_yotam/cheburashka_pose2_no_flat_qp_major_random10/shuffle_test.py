from __future__ import print_function, division

import random
import numpy

N = 3000
sub_N = N//10
X = numpy.arange(N)

python_shuffle_hist = numpy.zeros( N, dtype = int )
numpy_shuffle_hist  = numpy.zeros( N, dtype = int )
numpy_perm_hist     = numpy.zeros( N, dtype = int )

num_trials = 1000
for trial in range( num_trials ):
    if trial % 100 == 0: print( "Trial:", trial )
    
    ## Generate a random subset via python's built-in random.shuffle
    subX = X.copy()
    random.shuffle( subX )
    subX = subX[:sub_N]
    python_shuffle_hist[ subX ] += 1
    
    ## Generate a random subset via numpy's random.shuffle / random.permutation
    subX = numpy.random.permutation( X.copy() )[:sub_N]
    numpy_perm_hist[ subX ] += 1
    
    subX = X.copy()
    numpy.random.shuffle( subX )
    subX = subX[:sub_N]
    numpy_shuffle_hist[ subX ] += 1

numpy.set_printoptions( linewidth = 10000 )

print( "python random.shuffle histogram:" )
print( python_shuffle_hist )
print()
print( "numpy.random.shuffle histogram:" )
print( numpy_shuffle_hist )
print()
print( "numpy.random.permutation histogram:" )
print( numpy_perm_hist )
