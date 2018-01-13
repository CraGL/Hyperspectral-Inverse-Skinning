from __future__ import print_function, division

from numpy import *
import sys

print( "M1" )
M1 = loadtxt( sys.argv[1] )
print( "M2" )
M2 = loadtxt( sys.argv[2] )
print( "max abs difference:", abs( M1 - M2 ).max() )
