from __future__ import print_function, division

from math import *
import numpy
import numpy as np

def closest_rotation_3D( M ):
    '''
    Given a 3x3 matrix M, returns the closest pure rotation matrix.
    '''
    
    ## From: https://mathoverflow.net/questions/86539/closest-3d-rotation-matrix-in-the-frobenius-norm-sense
    
    U, S, Vt = np.linalg.svd( M )
    
    R = np.dot( U, Vt )
    if np.linalg.det( R ) < 0:
        U[:,2] = -U[:,2]
        R = np.dot( U, Vt )
    
    return R

def closest_rotations_R12( v ):
    '''
    Given a vector of affine transformation coefficients `v` obtained as in the packing
    used by flat_intersection.py, returns the same vector with each transformation
    projected to the closest rotation in 3D.
    
    The packing used by flat_intersection.py is a row-major flattening of the data
    as pose-by-three-by-four.
    '''
    
    v = np.asarray( v )
    assert len( v.shape ) == 1
    assert v.shape[0] % 12 == 0
    
    Ms = np.vsplit( v.reshape( ( -1, 4 ), order = 'C' ), v.shape[0]//12 )
    for M in Ms:
        M[:,:3] = closest_rotation_3D( M[:,:3] )
    result = np.vstack( Ms ).ravel( order = 'C' )
    return result

def test_closest_rotation_3D():
    ## Identity should remain unchanged.
    M = np.eye(3)
    R = closest_rotation_3D( M )
    numpy.testing.assert_allclose( M, R )
    
    ## The identity with a reflection should be closest to the identity.
    M = np.eye(3)
    M[2,2] = -1
    R = closest_rotation_3D( M )
    numpy.testing.assert_allclose( np.eye(3), R )
    
    ## A different rotation should stay that way.
    M = np.eye(3)
    theta = pi/10.
    M[0,0] = M[1,1] = cos(theta)
    M[0,1] = sin(theta)
    M[1,0] = -sin(theta)
    R = closest_rotation_3D( M )
    numpy.testing.assert_allclose( M, R )
    
    ## Scaling the rotation by a positive number shouldn't change the closest rotation.
    R = closest_rotation_3D( M*0.1 )
    numpy.testing.assert_allclose( M, R )
    R = closest_rotation_3D( M*1.5 )
    numpy.testing.assert_allclose( M, R )

def test_closest_rotation_R12():
    ## Identity should remain unchanged.
    M = [ np.eye(3) ] * 10
    Ms = np.append( np.vstack( M ), np.ones(len(M)*3).reshape( -1, 1 ), axis = 1 ).ravel( order = 'C' )
    Rs = closest_rotations_R12( Ms )
    numpy.testing.assert_allclose( Ms, Rs )

def main():
    print( "Debug the following by running: python -m pytest --pdb rigid.py" )
    import pytest
    pytest.main([__file__])
    # test_closest_rotation_3D()

if __name__ == '__main__':
    main()
