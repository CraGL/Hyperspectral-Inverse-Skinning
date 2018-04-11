from __future__ import print_function, division

import numpy as np
import scipy.linalg
import scipy.optimize
from trimesh import TriMesh
# import includes
import time



'''
Command:
python local_subspace_recover.py ./datas/cube4_copy/cube.obj ./datas/cube4_copy/poses-1/cube-*.obj -s 1e-15 -t 1e-4 -v 1  -o ./datas/cube4_copy/qs.txt -rand "euclidian" -ssize 30
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
        v0_center = V0[0]
        v1_center = V1[0]
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


def find_subspace_intersections(rest_pose, other_poses, version, method = None, use_pseudoinverse = None, propagate = None, random_sample_close_vertex="none", candidate_num=120, sample_num=10, vertices_num=40, precomputed_geodesic_distance=None):
    
    
    if propagate is None: propagate = False
    
    if method is None:
        method = "vertex"
    
    mesh0=rest_pose
    
    mesh1_list=other_poses
    pose_num=len(mesh1_list)
    
    vertices0=np.hstack((np.asarray(mesh0.vs),np.ones((len(mesh0.vs),1))))
    ## Stack all poses horizontally.
    vertices1 = np.hstack( [ mesh.vs for mesh in mesh1_list ] )
    
    if random_sample_close_vertex=="euclidian":
        data=vertices0[:,:3]
        vertices_pairwise_distance=np.sqrt(np.square(data.reshape((1,-1,3))-data.reshape((-1,1,3))).sum(axis=-1))
        np.random.seed(1)
    elif random_sample_close_vertex=="geodesic":
        import geodesic
        compute_distance=geodesic.GeodesicDistanceComputation(np.asarray(mesh0.vs), np.asarray(mesh0.faces))
        vertices_pairwise_distance=[]
        for i in range(len(mesh0.vs)):
            vertices_pairwise_distance.append(compute_distance(0))
        vertices_pairwise_distance=np.asarray(vertices_pairwise_distance)
        print (vertices_pairwise_distance.shape)
        np.random.seed(1)
    elif random_sample_close_vertex=="precomputed-geodesic":
        if precomputed_geodesic_distance is None:
            print ("Wrong!")
        else:
            vertices_pairwise_distance=precomputed_geodesic_distance
        
        np.random.seed(1)
        
        
    
    
    scale=find_scale(vertices0[:,:3])
    q_space=[]
    errors=[]
    smallest_singular_values=[]

    for i in range(len(vertices1)):
        
        if random_sample_close_vertex=="none":
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
        
        
        else:
            distance_to_others = vertices_pairwise_distance[i,:]
            sorted_distance_indices=np.argsort(distance_to_others)
            temp_q_space=[]
            temp_errors=[]
            temp_smallest_singular_values=[]
            for sample in range(sample_num):
                
                random_3_extra_vertices_indices=(np.random.random(vertices_num)*candidate_num).round().astype(np.int32)
                indices=np.asarray(sorted_distance_indices[random_3_extra_vertices_indices])
            
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
                temp_smallest_singular_values.append( ssv )

                assert q is not None
                # if q is not None:
                temp_q_space.append(q)
                temp_errors.append(np.sqrt(np.maximum(cost/(pose_num*scale*scale), 1e-30)))
             
            
            minimum_cost_ind=np.argmin(np.asarray(temp_errors))
            q_space.append(temp_q_space[minimum_cost_ind])
            errors.append(temp_errors[minimum_cost_ind])
            smallest_singular_values.append( temp_smallest_singular_values[minimum_cost_ind] )



    q_space=np.asarray(q_space)
    errors=np.asarray(errors)
    smallest_singular_values=np.asarray(smallest_singular_values)
    
    
    if propagate:
        something_changed = False
        ## We are going to assume triangles.
        assert mesh0.faces.shape[1] == 3
        def mag( x ): return np.dot( x,x )
        while True:
            print( "Propagating..." )
            something_changed = False
            for face in mesh0.faces:
                face_qs = q_space[ face ]
                
                for i in range(3):
                    # v = np.append( mesh.vs[ face[i] ], [1] )
                    v = vertices0[ face[i] ]
                    vprime = vertices1[ face[i] ]
                    
                    qs = [ q_space[i], q_space[(i+1)%3], q_space[(i+2)%3] ]
                    qvs = [
                        np.dot( v, q.reshape( 4, -1 ) ).reshape( vprime.shape ) for q in qs
                        ]
                    ## TODO Q: Should we add a little epsilon to qv_costs[1,2] so that
                    ## if the error is the same, it keeps the old one?
                    qv_costs = [
                        mag( qv - vertices1[ face[i] ] ) for qv in qvs
                        ]
                    
                    best_q_index = np.argmin( qv_costs )
                    if best_q_index != 0:
                        print( "Wow! Our neighbor had a better q. Changing." )
                        print( "Old diff:", qv_costs[0] )
                        print( "New diff:", qv_costs[best_q_index] )
                        print( "Old error:", errors[i] )
                        print( "New error:", errors[(i+best_q_index)%3] )
                        q_space[i] = qs[best_q_index]
                        ## Copy the error, too, even through that's not quite right.
                        ## TODO: Recompute that actual error for i and its neighbors.
                        errors[i] = errors[(i+best_q_index)%3]
                        something_changed = True
            
            if not something_changed: break
    
    # print (len(q_space))
    # print (max(errors))
    # print (len(q_space[errors<transformation_threshold]))
    # return q_space[errors<transformation_threshold]
    # return q_space[np.logical_and( errors<args.transformation_threshold, smallest_singular_values>=args.svd_threshold )]
    assert len( q_space ) == len( mesh0.vs )
    return q_space, errors, smallest_singular_values


def E_RMS_kavan2010( gt, data, scale=1.0):
    ### for vertex error
    E_RMS_kavan2010 = 1000*np.linalg.norm( gt.ravel() - data.ravel() )*2.0/np.sqrt(len(gt.ravel())*scale*scale) ## 3*pose_num*vs_num, and scale!
    return E_RMS_kavan2010

def vertex_reconstruction_error(vertices0, vertices1, q_space, scale=1.0):
    gt_vertices=[]
    reconstructed_vertices=[]
    pose_num=vertices1.shape[1]//3

    for i in range(len(vertices1)):
        v0=vertices0[i].reshape((1,-1))
        v1=vertices1[i].reshape((1,-1))

        v0_expand=np.zeros((3*pose_num,12*pose_num))
        for j in range(pose_num):
            for k in range(3):
                v0_expand[j*3+k, (j*3+k)*4:(j*3+k)*4+4]=v0.ravel()[4*j:4*j+4]

        t=q_space[i]
        v1_reconstructed=v0_expand.dot(t).ravel()
        reconstructed_vertices.append(v1_reconstructed)
        gt_vertices.append(v1.ravel())

    return E_RMS_kavan2010(np.asarray(gt_vertices), np.asarray(reconstructed_vertices), scale=scale)

def transformation_matrix_error(gt, data):
    diff=abs(gt-data).ravel()
    rmse=np.sqrt(np.square(gt-data).sum()/len(gt.ravel()))
    return [max(diff), min(diff), np.median(diff), np.mean(diff), rmse]
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser( description='Generate a set of transformations from a set of poses by finding local subspace intersections. Saves data as text where each line is ( 3-by-4-by-num_poses ).ravel()' )
    parser.add_argument( 'rest_pose', type=str, help='Path to rest pose (OBJ)' )
    parser.add_argument( 'other_poses', type=str, nargs='+', help='Paths to other poses (OBJ)')
    parser.add_argument( '--svd_threshold', '-s', type=float, help='Threshold for determining a singular vertex neighborhood (flat).' )
    parser.add_argument( '--transformation_threshold', '-t', type=float, help='Threshold for determining whether the subspaces intersect.' )
    parser.add_argument( '--transformation_percentile', '-p', type=float, help='Percentile for choosing best subspace intersections.' )
    parser.add_argument( '--version', '-v', type=int, help='0 means basic least square linear solver. 1 means constrained least square' )
    parser.add_argument( '--method', '-m', type=str, choices=["vertex","nullspace"], help='vertex: minimize transformed vertex error (default). nullspace: minimize distance to 3p-dimensional flats.' )
    parser.add_argument( '--propagate', action='store_true', help="If this flag is passed, then transformations will be propagated to neighbors if it helps." )
    parser.add_argument( '--pinv', type=bool, help='If True, use the pseudoinverse to solve the systems.' )
    parser.add_argument( '--print-all', type=bool, help='If True, prints all transformations, all costs, and all smallest singular values. Ignored if --out is specified.' )
    parser.add_argument( '--out', '-o', type=str, help='Path to store the result (prints to stdout if not specified).' )
    parser.add_argument( '--out-errors', type=str, help='Path to store the resulting cost.' )
    parser.add_argument( '--out-ssv', type=str, help='Path to store the smallest singular values.' )
    
    parser.add_argument( '--random-sample-method', '-rand', type=str, help=""" 'none' means one ring subspace intersection, others are 'euclidian', 'geodesic', 'precomputed-geodesic' """)
    parser.add_argument( '--subset-size', '-ssize', type=int, default=48, help='should smaller than 100, because search range is defalut to be 100' )
    parser.add_argument( '--precomputed-geodesic-distance-path', '-pre', type=str, help='Path to store the precomputed geodesic pairwise distance.' )
    parser.add_argument( '--save-dmat', '-dmat', type=str, help='save transformations as dmat' )

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

    if args.random_sample_method=="precomputed-geodesic":
        import gzip
        # f=gzip.GzipFile(base_dir+'/pairwise_geodesic_distance.npy.gz', "r")
        f=gzip.GzipFile(args.precomputed_geodesic_distance_path, "r")

        vertices_pairwise_distance=np.load(f)
        print (vertices_pairwise_distance.shape)
    else:
        vertices_pairwise_distance=None

    if args.random_sample_method=="none": ### no need sample
        sampled_subset_size=3 ## not useful
    else:
        sampled_subset_size=args.subset_size


    start_time = time.time()
    qs, errors, smallest_singular_values = find_subspace_intersections( rest_pose, other_poses, args.version, method = args.method, use_pseudoinverse = args.pinv, propagate = args.propagate, random_sample_close_vertex=args.random_sample_method, vertices_num=sampled_subset_size, precomputed_geodesic_distance=vertices_pairwise_distance)
    end_time = time.time()
    print( "... Finished generating transformations." )
    print( "Finding subspace intersection duration (minutes):", (end_time-start_time)/60 )

    if args.out is None:
        np.set_printoptions( precision = 24, linewidth = 2000 )
        def save_one( path, M ):
            print( repr( M ) )
    else:
        def save_one( path, M ):
            np.savetxt( path, M )
            print( "Saved to path:", path )
            
    if args.save_dmat is not None:
        assert args.out is not None
        dmat = qs.reshape(qs.shape[0], -1, 12)
        dmat = np.swapaxes(dmat, 0, 1)
        import os, sys
        sys.path.append("..")
        from InverseSkinning import format_loader as fl
        name = os.path.join(args.save_dmat, os.path.splitext( os.path.split(args.out)[1] )[0])
        for i in range(len(dmat)):
            idx = "0000"
            idx = idx[:-len(str(i))] + str(i)
            path = name + "-" + idx + ".DMAT"
            data = dmat[i].reshape(dmat[i].shape[0], 3, 4)
            data = np.swapaxes(data, 1, 2)
            data = data.reshape(data.shape[0], 12)
            fl.write_DMAT(path, data)
            print("save transformations to ", path)
    
    if args.print_all:
        print( "# qs" )
        save_one( args.out, qs )
        print( "# transformation errors" )
        save_one( args.out_errors, errors )
        print( "# smallest_singular_values" )
        save_one( args.out_ssv, smallest_singular_values )
    elif args.transformation_threshold is not None:
        qs_to_save = qs[np.logical_and( errors<args.transformation_threshold, smallest_singular_values>=args.svd_threshold )]
        print( "About to save qs with dimension:", qs_to_save.shape )
        save_one( args.out, qs_to_save )
    elif args.transformation_percentile is not None:
        qs_to_save = qs[ smallest_singular_values>=args.svd_threshold ]
        topN = int( len( qs_to_save )*args.transformation_percentile/100 )
        qs_to_save = qs_to_save[ np.argsort( errors[ smallest_singular_values>=args.svd_threshold ] )[:topN], : ]
        print( "About to save qs with dimension:", qs_to_save.shape )
        save_one( args.out, qs_to_save )
    else:
        raise RuntimeError( "No combination of --print-all, --transformation_threshold, or --transformation_percentile was given." )
