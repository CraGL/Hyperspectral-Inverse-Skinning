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


def func(q, V0, V1):
    pose_num=V0.shape[1]//4
    obj=np.zeros(0)
    for i in range(len(V0)):
        v0=V0[i]
        v1=V1[i]
        v0_expand=np.zeros((3*pose_num,12*pose_num))
        for j in range(pose_num):
            for k in range(3):
                v0_expand[j*3+k, (j*3+k)*4:(j*3+k)*4+4]=v0[4*j:4*j+4]

##### version 1
        Null_matrix=null_space_matrix(v0_expand)
        T=np.zeros(12*pose_num).reshape((pose_num,3,4))
        for j in range(pose_num):
            T[j,:,:-1]=np.identity(3)
            T[j,:,-1]=v1[3*j:3*j+3]-v0[4*j:4*j+3]
        T=T.ravel()
        obj=np.concatenate((obj,(q-(T + Null_matrix.T.dot(np.dot(Null_matrix,q-T)))).ravel()))
    
#### version 2
#         obj=np.concatenate((obj, (np.dot(v0_expand, q)-v1).ravel()))
        
    return obj


        
def compute_I_Projection_matrix(V0):
    pose_num=V0.shape[1]//4
    sum_I_P=np.zeros((12*pose_num,12*pose_num))
    Identity=np.identity(12*pose_num)
    for i in range(len(V0)):
        v0=V0[i]
        v0_expand=np.zeros((3*pose_num,12*pose_num))
        for j in range(pose_num):
            for k in range(3):
                v0_expand[j*3+k, (j*3+k)*4:(j*3+k)*4+4]=v0[4*j:4*j+4]
                
        Null_matrix=null_space_matrix(v0_expand)
        
        P=Null_matrix.T.dot(Null_matrix)
        sum_I_P+=(Identity-P)
    s= scipy.linalg.svd(sum_I_P, compute_uv=False)
    return s
        
        
def solve(q0, V0, V1):
    res=scipy.optimize.least_squares(func, q0, args=(V0, V1),jac='3-point', method='trf')
    q=res["x"]
    cost=res["cost"]
    return q, cost


def solve_directly(V0, V1, method, version=0):
    pose_num=V0.shape[1]//4
    left=np.zeros((12*pose_num, 12*pose_num))
    right=np.zeros((12*pose_num))
    constant=0.0
    v_expand_center=None
    v1_center=None
    for i in range(len(V0)):
        v0=V0[i]
        v1=V1[i]
        v0_expand=np.zeros((3*pose_num,12*pose_num))
        for j in range(pose_num):
            for k in range(3):
                v0_expand[j*3+k, (j*3+k)*4:(j*3+k)*4+4]=v0[4*j:4*j+4]
        if i==0:
            v_expand_center=v0_expand.copy()
            v1_center=v1.copy()

        if method == "nullspace":
            #### version 1
            Identity=np.identity(12*pose_num)
            Null_matrix=null_space_matrix(v0_expand)
            P=Null_matrix.T.dot(Null_matrix)
            T=np.zeros(12*pose_num).reshape((pose_num,3,4))
            for j in range(pose_num):
                T[j,:,:-1]=np.identity(3)
                T[j,:,-1]=v1[3*j:3*j+3]-v0[4*j:4*j+3]
            T=T.ravel()

            A = (Identity-P).T.dot((Identity-P))
            left+=A
            right+=A.dot(T)
            constant+=T.dot(A).dot(T)

            
        elif method == "vertex":
            ##### version 2
            left+=v0_expand.T.dot(v0_expand)
            right+=v0_expand.T.dot(v1)
            constant+=v1.T.dot(v1)
        
        else:
            raise RuntimeError
    
    if version==0:
        x=scipy.linalg.solve(left,right)
    elif version==1:
        new_left=np.hstack((left, v_expand_center.T))
        temp=np.hstack((v_expand_center, np.zeros((3*pose_num, 3*pose_num))))
        new_left=np.vstack((new_left, temp))
        new_right=np.concatenate((right, v1_center))
        x_full=scipy.linalg.solve(new_left,new_right)
        x=x_full[:12*pose_num]

    return x, (x.T.dot(left).dot(x)-2*right.T.dot(x)+constant).squeeze()
    


def find_scale(Vertices):
    Dmin=Vertices.min(axis=0)
    Dmax=Vertices.max(axis=0)
    D=Dmax-Dmin
    scale=np.sqrt(D.dot(D))
    return scale


def find_subspace_intersections( *args, **kwargs ):
    #### read obj file into mesh
    rest_pose_name=args[0]
    other_poses_name=args[1]
    svd_threshold=args[2]
    transformation_threshold=args[3]
    version=args[4]
    
    mesh0=TriMesh.FromOBJ_FileName(rest_pose_name[0])
    mesh1_list=[]
    pose_num=len(other_poses_name)

    for i in range(pose_num):
        mesh1_list.append(TriMesh.FromOBJ_FileName(other_poses_name[i]))


    vertices0=vertices0_temp=np.hstack((np.asarray(mesh0.vs),np.ones((len(mesh0.vs),1))))
    vertices1=vertices1_temp=np.asarray(mesh1_list[0].vs)
    for i in range(1,len(mesh1_list)):
        vertices0=np.hstack((vertices0, vertices0_temp))
        vertices1=np.hstack((vertices1, np.asarray(mesh1_list[i].vs)))
    
    scale=find_scale(vertices0[:,:3])
    q_space=[]
    errors=[]

    for i in range(len(vertices1)):
        indices = mesh0.vertex_vertex_neighbors(i)
        indices=np.asarray(indices)
        
        if len(indices)>=3:
            v0=vertices0[i].reshape((1,-1))
            v0_neighbor=vertices0[indices,:]
            v1=vertices1[i].reshape((1,-1))
            v1_neighbor=vertices1[indices,:]
            
            V0=np.vstack((v0, v0_neighbor))
            V1=np.vstack((v1, v1_neighbor))

            s=compute_I_Projection_matrix(V0)
            if s[-1] < svd_threshold:
                continue

            #### solve using optimization 
            # q0=np.random.random((12*pose_num,))
            # q,cost=solve(q0, V0, V1)
            # q_space.append(q)
            # errors.append(cost)

            #### solve directly
            q,cost=solve_directly(V0, V1, "vertex", version)

            if q is not None:
                q_space.append(q)
                errors.append(np.sqrt(max(cost/(pose_num*scale*scale), 1e-30)))


    q_space=np.asarray(q_space)
    errors=np.asarray(errors)

    # print (len(q_space))
    # print (max(errors))
    # print (len(q_space[errors<transformation_threshold]))
    return q_space[errors<transformation_threshold]



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser( description='Generate a set of transformations from a set of poses by finding local subspace intersections. Saves data as text where each line is ( 3-by-4-by-num_poses ).ravel()' )
    parser.add_argument( 'rest_pose', type=str, nargs=1, help='Path to rest pose (OBJ)' )
    parser.add_argument( 'other_poses', type=str, nargs='+', help='Paths to other poses (OBJ)')
    parser.add_argument( '--svd_threshold', '-s', type=float, help='Threshold for determining a singular vertex neighborhood (flat).' )
    parser.add_argument( '--transformation_threshold', '-t', type=float, help='Threshold for determining whether the subspaces intersect.' )
    parser.add_argument( '--version', '-v', type=int, help='0 means basic least square linear solver. 1 means constrained least square' )
    parser.add_argument( '--out', '-o', type=str, help='Path to store the result (prints to stdout if not specified).' )

    args = parser.parse_args()

    print( "Generating transformations..." )
    start_time = time.time()
    qs = find_subspace_intersections( args.rest_pose, args.other_poses, args.svd_threshold, args.transformation_threshold, args.version )
    print( "... Finished generating transformations." )
    print( "Finding subspace intersection costs: ", time.time()-start_time )

    if args.out is None:
        np.set_printoptions( precision = 24, linewidth = 2000 )
        print( repr( qs ) )
    else:
        np.savetxt( args.out, qs )
        print( "Saved:", args.out )
