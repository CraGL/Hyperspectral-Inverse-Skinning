from __future__ import print_function, division

import numpy as np
import scipy.linalg
import scipy.optimize
from trimesh import TriMesh
import time



'''
Command:
python local_subspace_recover.py ./datas/cube4_copy/cube.obj ./datas/cube4_copy/poses-1/cube-*.obj -s 1e-15 -t 1e-4 -v 1  -o ./datas/cube4_copy/qs.txt
'''

def null_space_matrix(A):
    pose_num=A.shape[1]//12
    u, s, vh = scipy.linalg.svd(A)
    null_space=vh[3*pose_num:,:]
    return null_space

def null_space_projection_matrix(v):
    v = v.squeeze()
    assert v.shape == (4,)
    
    # vnorm = np.sqrt( np.dot( v,v ) )
    # row = v / vnorm
    # np.outer( row, row )
    
    ## The projection matrix onto the null space is:
    # P_row = np.outer( row, row )
    ## We can write the outer product a little more efficiently:
    P = np.outer( v, v / np.dot( v, v ) )
    
    ## The projection matrix onto the row space is (I-P_null)
    # P = np.outer( -row, row )
    # P[ np.diag_indices_from(P) ] += 1
    
    return P

def v_to_vbar( v, pose_num ):
    '''
    Given:
        v: a 4-vector
        pose_num: an integer
    
    Returns
        The kronecker product of the identity matrix of size 3*pose_num with v as a row matrix:
        np.kron( np.eye( 3*pose_num ), v.reshape(1,-1) )
    
    This is the Vbar and v_expand used in the code.
    '''
    result = np.zeros((3*pose_num,12*pose_num))
    for j in range(pose_num):
        for k in range(3):
            result[j*3+k, (j*3+k)*4:(j*3+k)*4+4]=v
    return result

def func(q, V0, V1):
    pose_num=V1.shape[1]//3
    obj=np.zeros(0)
    for i in range(len(V0)):
        v0=V0[i]
        v1=V1[i]
        v0_expand=v_to_vbar( v0, pose_num )

##### version 1
        Null_matrix=null_space_matrix(v0_expand)
        T=np.zeros(12*pose_num).reshape((pose_num,3,4))
        for j in range(pose_num):
            T[j,:,:-1]=np.identity(3)
            T[j,:,-1]=v1[3*j:3*j+3]-v0[:3]
        T=T.ravel()
        obj=np.concatenate((obj,(q-(T + Null_matrix.T.dot(np.dot(Null_matrix,q-T)))).ravel()))
        
#### version 2
#         obj=np.concatenate((obj, (np.dot(v0_expand, q)-v1).ravel()))
        
    return obj


'''
def compute_I_Projection_matrix(V0,pose_num):
    assert V0.shape[1] == 4
    sum_I_P=np.zeros((12*pose_num,12*pose_num))
    Identity=np.identity(12*pose_num)
    for i in range(len(V0)):
        v0=V0[i]
        v0_expand=v_to_vbar( v0, pose_num )
                
        Null_matrix=null_space_matrix(v0_expand)
        
        P=Null_matrix.T.dot(Null_matrix)
        sum_I_P+=(Identity-P)
    s= scipy.linalg.svd(sum_I_P, compute_uv=False)
    s2 = compute_I_Projection_matrix_smallest_singular_value(V0,pose_num)
    sum_I_P2 = np.sum( [null_space_projection_matrix(v) for v in V0], axis = 0 )
    return s
'''

def compute_I_Projection_matrix_smallest_singular_value(V0,pose_num):
    ## We want to compute the sum of 
    P = np.zeros( ( 4,4 ) )
    for v in V0: P += null_space_projection_matrix(v)
    ## Get the smallest singular value of P
    ssv = np.linalg.norm( P, ord = -2 )
    return ssv

compute_I_Projection_matrix = compute_I_Projection_matrix_smallest_singular_value

        
def solve(q0, V0, V1):
    res=scipy.optimize.least_squares(func, q0, args=(V0, V1),jac='3-point', method='trf')
    q=res["x"]
    cost=res["cost"]
    return q, cost


def solve_directly(V0, V1, method, version=0, use_pseudoinverse = None):
    if use_pseudoinverse is None: use_pseudoinverse = False
    pose_num=V1.shape[1]//3
    left=np.zeros((4,4))
    right=np.zeros((4,3*pose_num))
    constant=0.0
    
    if method == "nullspace":
        ## Allocate the translation matrices and fill in the upper-left portion with
        ## the identity matrix.
        Tidentity=np.zeros(12*pose_num).reshape((pose_num,3,4))
        for j in range(pose_num):
            Tidentity[j,:,:-1]=np.identity(3)
    
    for i in range(len(V0)):
        v0=V0[i].reshape(-1,1)
        v1=V1[i].reshape(-1,1)
        
        if method == "nullspace":
            #### version 1
            for j in range(pose_num):
                Tidentity[j,:,-1]=( v1[3*j:3*j+3]-v0[:3] ).squeeze()
            T=Tidentity.ravel()

            I_P = null_space_projection_matrix(v0)
            # A = (I_P).T.dot((I_P))
            ## A projection matrix times itself is the same.
            A = I_P
            left+=A
            
            # import scipy.linalg
            # AT = scipy.linalg.block_diag( *( [A]*(3*pose_num) ) ).dot( T )
            # assert abs(np.dot( A, T.reshape( ( 4,-1 ), order = 'F' ) ).ravel( order = 'F' ) - AT).max() < 1e-15
            AT = np.dot( A, T.reshape( ( 4,-1 ), order = 'F' ) )
            
            right+=AT
            constant+=T.dot(AT.ravel( order = 'F' ))

            
        elif method == "vertex":
            ##### version 2
            left     += v0.dot( v0.T )
            right    += v0.dot( v1.T )
            constant += v1.T.dot(v1).squeeze()
        
        else:
            raise RuntimeError
    
    ssv = np.linalg.norm( left, ord = -2 )
    if ssv < 1e-10: use_pseudoinverse = True
    
    if version==0:
        if use_pseudoinverse:
            ## Reshape the right-hand-side in column-major order (F).
            x=np.linalg.pinv(left).dot( right ).ravel( order='F' )
        else:
            ## Reshape the right-hand-side in column-major order (F).
            x=scipy.linalg.solve( left, right ).ravel( order='F' )
    elif version==1:
        new_left = np.zeros( ( 5,5 ) )
        new_left[:4,:4] = left
        new_left[:-1,-1] = v0_center.T.squeeze()
        new_left[-1,:-1] = v0_center.squeeze()
        new_right = np.vstack( ( right, v1_center.reshape( (1,-1) ) ) )
        if use_pseudoinverse:
            x_full=np.linalg.pinv(new_left).dot(new_right)
        else:
            x_full=scipy.linalg.solve(new_left,new_right)
        ## Reshape the right-hand-side in column-major order (F).
        x=x_full[:-1].ravel( order='F' )

    y = x.reshape( (4,-1), order='F' )
    cost = (y*left.dot(y)).sum()-2*(right*y).sum()+constant
    return x, cost, ssv

def find_scale(Vertices):
    Dmin=Vertices.min(axis=0)
    Dmax=Vertices.max(axis=0)
    D=Dmax-Dmin
    scale=np.sqrt(D.dot(D))
    return scale


def find_subspace_intersections( rest_pose, other_poses, version, method = None, use_pseudoinverse = None, propagate = None ):
    
    if propagate is None: propagate = False
    
    if method is None:
        method = "vertex"
    
    mesh0=rest_pose
    
    mesh1_list=other_poses
    pose_num=len(mesh1_list)
    
    vertices0=np.hstack((np.asarray(mesh0.vs),np.ones((len(mesh0.vs),1))))
    ## Stack all poses horizontally.
    vertices1 = np.hstack( [ mesh.vs for mesh in mesh1_list ] )
    
    scale=find_scale(vertices0[:,:3])
    q_space=[]
    errors=[]
    smallest_singular_values=[]

    for i in range(len(vertices1)):
        indices = mesh0.vertex_vertex_neighbors(i)
        indices=np.asarray(indices)
        
        ## We want everything, we'll use the pseudoinverse.
        # if len(indices)>=3:
        v0=vertices0[i].reshape((1,-1))
        v0_neighbor=vertices0[indices,:]
        v1=vertices1[i].reshape((1,-1))
        v1_neighbor=vertices1[indices,:]
        
        V0=np.vstack((v0, v0_neighbor))
        V1=np.vstack((v1, v1_neighbor))

        #### solve directly
        q,cost,ssv=solve_directly(V0, V1, method = method, version = version, use_pseudoinverse = use_pseudoinverse)
        smallest_singular_values.append( ssv )

        assert q is not None
        # if q is not None:
        q_space.append(q)
        errors.append(np.sqrt(max(cost/(pose_num*scale*scale), 1e-30)))


    q_space=np.asarray(q_space)
    errors=np.asarray(errors)
    smallest_singular_values=np.asarray(smallest_singular_values)
    
    
    if propagate:
        something_changed = False
        ## We are going to assume triangles.
        assert mesh0.faces.shape[1] == 3
        while True:
            for face in mesh0.faces:
                face_qs = q_space[ face ]
                
                for i in range(3):
                    v = np.append( mesh.vs[ face[i] ], [1] )
                    
                    v_transformed_own_q = np.dot( v, qs[i].reshape( 4, -1 ) ).reshape( vprime.shape )
                    v_transformed_next_q = np.dot( v, qs[(i+1)%3].reshape( 4, -1 ) ).reshape( vprime.shape )
                    v_transformed_prev_q = np.dot( v, qs[(i-1)%3].reshape( 4, -1 ) ).reshape( vprime.shape )
                    
                    v_transformed_own_q_cost = v_transformed_own_q - vertices1[ face[i] ]
                    
            
            v_transformed = np.dot( v, transformation.reshape( 4, -1 ) ).reshape( vprime.shape )
            
            if not something_changed: break

    # print (len(q_space))
    # print (max(errors))
    # print (len(q_space[errors<transformation_threshold]))
    # return q_space[errors<transformation_threshold]
    # return q_space[np.logical_and( errors<args.transformation_threshold, smallest_singular_values>=args.svd_threshold )]
    assert len( q_space ) == len( mesh0.vs )
    return q_space, errors, smallest_singular_values



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser( description='Generate a set of transformations from a set of poses by finding local subspace intersections. Saves data as text where each line is ( 3-by-4-by-num_poses ).ravel()' )
    parser.add_argument( 'rest_pose', type=str, help='Path to rest pose (OBJ)' )
    parser.add_argument( 'other_poses', type=str, nargs='+', help='Paths to other poses (OBJ)')
    parser.add_argument( '--svd_threshold', '-s', type=float, help='Threshold for determining a singular vertex neighborhood (flat).' )
    parser.add_argument( '--transformation_threshold', '-t', type=float, help='Threshold for determining whether the subspaces intersect.' )
    parser.add_argument( '--version', '-v', type=int, help='0 means basic least square linear solver. 1 means constrained least square' )
    parser.add_argument( '--method', '-m', type=str, choices=["vertex","nullspace"], help='vertex: minimize transformed vertex error (default). nullspace: minimize distance to 3p-dimensional flats.' )
    parser.add_argument( '--pinv', type=bool, help='If True, use the pseudoinverse to solve the systems.' )
    parser.add_argument( '--print-all', type=bool, help='If True, prints all transformations, all costs, and all smallest singular values. Ignored if --out is specified.' )
    parser.add_argument( '--out', '-o', type=str, help='Path to store the result (prints to stdout if not specified).' )
    parser.add_argument( '--out-errors', type=str, help='Path to store the resulting cost.' )
    parser.add_argument( '--out-ssv', type=str, help='Path to store the smallest singular values.' )

    args = parser.parse_args()

    #### read obj files
    print( "Loading rest pose mesh:", args.rest_pose )
    rest_pose=TriMesh.FromOBJ_FileName(args.rest_pose)
    ## Make sure the mesh is storing arrays.
    rest_pose.vs = np.asarray( rest_pose.vs )
    rest_pose.faces = np.asarray( rest_pose.faces, dtype = int )

    print( "Loading", len( args.other_poses ), "other mesh poses..." )
    other_poses = [ TriMesh.FromOBJ_FileName( path ) for path in args.other_poses ]
    for mesh in other_poses:
        mesh.vs = np.asarray( mesh.vs )
        mesh.faces = np.asarray( mesh.faces, dtype = int )
    print( "...done." )

    print( "Generating transformations..." )
    start_time = time.time()
    qs, errors, smallest_singular_values = find_subspace_intersections( rest_pose, other_poses, args.version, method = args.method, use_pseudoinverse = args.pinv )
    end_time = time.time()
    print( "... Finished generating transformations." )
    print( "Finding subspace intersection duration (seconds): ", end_time-start_time )

    if args.out is None:
        np.set_printoptions( precision = 24, linewidth = 2000 )
        if args.print_all:
            print( "# qs" )
            print( repr( qs ) )
            print( "# transformation errors" )
            print( repr( errors ) )
            print( "# smallest_singular_values" )
            print( repr( smallest_singular_values ) )
        else:
            print( repr( qs[np.logical_and( errors<args.transformation_threshold, smallest_singular_values>=args.svd_threshold )] ) )
    elif args.print_all:
        np.savetxt( args.out, qs )
        print( "Saved:", args.out )
        
        if args.out_errors is not None:
            np.savetxt( args.out_errors, errors )
            print( "Saved:", args.out_errors )
        
        if args.out_ssv is not None:
            np.savetxt( args.out_ssv, smallest_singular_values )
            print( "Saved:", args.out_ssv )
    else:
        np.savetxt( args.out, qs[np.logical_and( errors<args.transformation_threshold, smallest_singular_values>=args.svd_threshold )] )
        print( "Saved:", args.out )
