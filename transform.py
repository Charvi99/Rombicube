import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as rot
from scipy.spatial import distance
from scipy.optimize import leastsq
from numpy import linalg as LA
import start_init as init

import transforms3d as tf3d


def rvecsTvecsToTransfMatrix(tvecs, rvecs):
    transform_matrixes = []
    new_rvec = []
    #print(tvecs)
    for i in range(len(tvecs)) :
        # print(tvecs[i][0])
        # print(tvecs[i][0][0])        
        transform_matrix = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1]],
                                    dtype=float)
        transform_matrix[0, 3] = tvecs[i][0][0]
        transform_matrix[1, 3] = tvecs[i][0][1]
        transform_matrix[2, 3] = tvecs[i][0][2]
        T = tvecs[i][0]
        R = cv2.Rodrigues(rvecs[i])[0]
        # Unrelated -- makes Y the up axis, Z forward
        R = R @ np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0,-1, 0],
        ])
        if 0 < R[1,1] < 1:
            # If it gets here, the pose is flipped.

            # Flip the axes. E.g., Y axis becomes [-y0, -y1, y2].
            R *= np.array([
                [ 1, -1,  1],
                [ 1, -1,  1],
                [-1,  1, -1],
            ])
            
            # Fixup: rotate along the plane spanned by camera's forward (Z) axis and vector to marker's position
            forward = np.array([0, 0, 1])
            tnorm = T / np.linalg.norm(T)
            axis = np.cross(tnorm, forward)
            angle = -2*math.acos(tnorm @ forward)
            R = cv2.Rodrigues(angle * axis)[0] @ R
        transform_matrix[:3, :3] = R
        rot_matrix = rot.from_matrix(R)
        new_rvec.append(rot_matrix.as_rotvec())
        transform_matrixes.append(transform_matrix)
    return transform_matrixes, new_rvec

def rvecsTvecsToTransfMatrix2(tvecs, rvecs):      
    transform_matrix = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 1]],
                                dtype=float)
    transform_matrix[0, 3] = tvecs[0]
    transform_matrix[1, 3] = tvecs[1]
    transform_matrix[2, 3] = tvecs[2]
    R = cv2.Rodrigues(rvecs)[0]

    transform_matrix[:3, :3] = R

    return transform_matrix
        
def rvecTvecToTransfMatrix(tvec, rvec):    
    # we need a homogeneous matrix but OpenCV only gives us a 3x3 rotation matrix
    transform_matrix = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 1]],
                                dtype=float)
    transform_matrix[:3, :3], _ = cv2.Rodrigues(rvec)
    transform_matrix[0, 3] = tvec[0]
    transform_matrix[1, 3] = tvec[1]
    transform_matrix[2, 3] = tvec[2]
    T = tvec[0]
    return transform_matrix


def centerAruco2(trans_matrix_array, offset_matrix):

    #print(aruco_index)
    transformed_aruco = []

    for i, aruco in enumerate(trans_matrix_array):
        index_of_aruco = aruco[0][0]
        #print(index_of_aruco)
        if index_of_aruco < 18:
    
            offset_of_aruco = np.linalg.inv(offset_matrix[index_of_aruco])
            transformed_aruco.append(np.matmul(aruco[1],offset_of_aruco))

    return transformed_aruco

def centerAruco3(trans_matrix_array, offset_matrix):
    transformed_aruco = []

    for i, aruco in enumerate(trans_matrix_array):
        index_of_aruco = aruco[0][0]
        #print(index_of_aruco)
        if index_of_aruco < 18:
            
            offset_of_aruco = np.linalg.inv(offset_matrix[index_of_aruco])
            # transformed_aruco.append(np.matmul(aruco[1],offset_of_aruco))
            transformed_aruco.append(fuseArucoRotation(np.array([aruco[1],offset_of_aruco])))

    return transformed_aruco
    
def fuseMatrix(trans_matrix_array):
    out_trans = np.eye(4,4)
    for aruco in trans_matrix_array:
        out_trans = np.matmul(out_trans,aruco)
    return out_trans

def removeBadCandidates(transform_matrix_centered,mtx,dist):
    max_distance = 0.1
    centers_R3 = transform_matrix_centered[:,0:3,3]
    projected_in_pix_sp,_ = cv2.projectPoints(centers_R3,np.zeros((3,1)),np.zeros((3,1)), mtx, dist) 

    mean = [np.mean(centers_R3[:,0]), np.mean(centers_R3[:,1]),np.mean(centers_R3[:,2])]

    good_indices = []
    for i in range(len(centers_R3)):
        counter = 0
        for j in range(3):
            if np.abs(mean[j] - centers_R3[i,j])<max_distance:
                counter = counter+1
        if counter == 3:
            good_indices.append(i)
    return transform_matrix_centered[good_indices, :, :] , 0 , 0 

    
    
    max_distance = 50
    centers_R3 = transform_matrix_centered[:,0:3,3]
    projected_in_pix_sp,_ = cv2.projectPoints(centers_R3,np.zeros((3,1)),np.zeros((3,1)), mtx, dist) 
    projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)
    distances = distance.cdist(centers_R3, centers_R3)
    distances_2 = distance.cdist(projected_in_pix_sp, projected_in_pix_sp)

    good_pairs = (distances_2 > 0) * (distances_2 < max_distance)

    good_indices = np.where(np.sum(good_pairs, axis=0) > 0)[0].flatten()

    if good_indices.shape[0] == 0 :
        print('good_indices is none, resetting')
        good_indices = np.array([0]) 

    return transform_matrix_centered[good_indices, :, :], centers_R3[good_indices, :], good_indices

def transMatrixToTvecRvec(transform_matrix):
    rvec = tf3d.axangles.mat2axangle(transform_matrix[:3, :3])
    rvec = np.array([rvec[0][0], rvec[0][1], rvec[0][2]])
    tvec = transform_matrix[0:3,3]
    return tvec, rvec

def slerp(v0, v1, t_array):
        # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0*v1)
    if (dot < 0.0):
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis,:] + t_array[:,np.newaxis]*(v1 - v0)[np.newaxis,:]
        result = result/np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0*t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])

def getQuaternion(trans_matrix):
    return tf3d.quaternions.mat2quat(trans_matrix[0:3,0:3])
def quaternionAreAlmostSame(quat1, quat2):
    return tf3d.quaternions.nearly_equivalent(quat1,quat2,rtol=0.05,atol=0.05)
def fuseArucoRotation(transform_matrix, output_ids=[], output_corners=[]):
    Tf_cam_ball = np.eye(4)

	#### using slerp interpolation for averaging rotations
    sum_tra = np.zeros(3,)
    quat_av = tf3d.quaternions.mat2quat(transform_matrix[0][0:3,0:3])
    for itr in range(transform_matrix.shape[0]-1):
        quat2 = tf3d.quaternions.mat2quat(transform_matrix[itr+1][0:3,0:3])
        quat_av = slerp(quat2,quat_av,[0.5])
        quat_av =quat_av.reshape(4,)

    Tf_cam_ball[0:3,0:3]=tf3d.quaternions.quat2mat(quat_av)


    for itr in range(transform_matrix.shape[0]):
        sum_tra = sum_tra + transform_matrix[itr][0:3,3]

    sum_tra = sum_tra/transform_matrix.shape[0]
    Tf_cam_ball[0:3,3] = sum_tra

    #APE_pose = APE(Tf_cam_ball, output_ids, output_corners)

    return 	Tf_cam_ball#, APE_pose


######################################################################################### 
######################################### APE ########################################### 
#########################################################################################
def RodriguesToTransf(x):
    '''
	Function to get a SE(3) transformation matrix from 6 Rodrigues parameters. NEEDS CV2.RODRIGUES()
	input: X -> (6,) (rvec,tvec)
	Output: Transf -> SE(3) rotation matrix
	'''
    x = np.array(x)
    x = x.reshape(6,)
    rot,_ = cv2.Rodrigues(x[0:3])
    trans =  np.reshape(x[3:6],(3,1))
    Transf = np.concatenate((rot,trans),axis = 1)
    Transf = np.concatenate((Transf,np.array([[0,0,0,1]])),axis = 0)
    return Transf


def LM_APE_Dodecapen(X,stacked_corners_px_sp, ids, flag=False):
    '''
	Function to get the objective function for APE step of the algorithm
	TODO: Have to put it in its respective class as a method (kind attn: Howard)
	Inputs: 
	X: (6,) array of pose parameters [rod_1, rod_2,rod_3,x,y,z]
	stacked_corners_px_sp = Output from Aruco marker detection. ALL the corners of the markers seen stacked in order 
	ids: int array of ids seen -- ids of faces seen
	Output: V = [4*M x 1] numpy array of difference between pixel distances
	'''
	# print(ids)
    k = np.load("sony_npy/calibration_matrix.npy")
    d = np.load("sony_npy/distortion_coefficients.npy")

    corners_in_cart_sp = np.zeros((ids.shape[0],4,3))
    Tf_cam_ball = RodriguesToTransf(X)
    for ii in range(ids.shape[0]):
        Tf_cent_face = init.init_offset_matrix(localization_unit_size=0.015)[int(ids[ii])]
        corners_in_cart_sp[ii,:,:] = Tf_cam_ball.dot(corners_3d(Tf_cent_face, 0.015)).T[:,0:3]

    corners_in_cart_sp = corners_in_cart_sp.reshape(ids.shape[0]*4,3)
    projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),
                                               k,d)

    projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)
    n,_=np.shape(stacked_corners_px_sp)
    V = LA.norm(stacked_corners_px_sp-projected_in_pix_sp, axis=1)

    if flag is False:
        return V

def corners_3d(tf_mat,m_s):
    '''
	Function to give coordinates of the marker corners and transform them using a given transformation matrix
	Inputs:
	tf_mat = transformation matrix between frames
	m_s = marker size-- edge lenght in mm
	Outputs:
	corn_pgn_f = corners in camara frame
	'''
    tf_mat = np.array(tf_mat)
    corn_1 = np.array([-m_s/2.0,  m_s/2.0, 0, 1])
    corn_2 = np.array([ m_s/2.0,  m_s/2.0, 0, 1])
    corn_3 = np.array([ m_s/2.0, -m_s/2.0, 0, 1])
    corn_4 = np.array([-m_s/2.0, -m_s/2.0, 0, 1])
    corn_mf = np.vstack((corn_1,corn_2,corn_3,corn_4))
    corn_pgn_f = tf_mat.dot(corn_mf.T)
    return corn_pgn_f

def APE(Tf_cam_ball, ids, corners):
        r_vec_ball,_ = cv2.Rodrigues(Tf_cam_ball[0:3,0:3])
        t_vec_ball = Tf_cam_ball[0:3,3]


        X_guess = np.append(r_vec_ball,np.reshape(t_vec_ball,(3,1))).reshape(6,1)
        pose_marker_without_opt = X_guess.T # not efficient. May have to change

        stacked_corners_px_sp =  np.reshape(np.asarray(corners),(ids[0].shape[0]*4,2))

		
        res = leastsq (LM_APE_Dodecapen,X_guess,Dfun=None, full_output=0, 
            col_deriv=0, ftol=1.49012e-6, xtol=1.49012e-4, gtol=0.0, 
            maxfev=1000, epsfcn=None, factor=1, diag=None,
            args = (stacked_corners_px_sp, ids, False)) 
#---------------------------------- returns res --------------------------------------------
		
        pose_marker_with_APE = np.reshape(res[0],(1,6)) 


def tipPosition(trans_mat):
    tip_mat = np.eye(4,4)
    tip_mat[2,3] = -0.185
    output = np.matmul(trans_mat,tip_mat)
    return output

def getAngles(trans_mat):
    R = rot.from_matrix(trans_mat[:3, :3])
    euler = R.as_euler('zxy', degrees=True)
    # euler[2] = abs(euler[2])
    return euler
    