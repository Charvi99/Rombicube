# my libs
import transform as trans
from start_init import Init
from visualisation import Visaliser

# other libs
import numpy as np
import cv2
import apriltag
import time

class RombiCube():
    def __init__(self,unit_size, marker_size, vis_enable = False):
        # load parameters for camera
        # self.camera_matrix = np.load("webcam_npy/calibration_matrix.npy")
        # self.distortion_matrix = np.load("webcam_npy/distortion_coefficients.npy")

        # self.camera_matrix = np.load("hq_calib_2000_1500_mes/calibration_matrix.npy")
        # self.distortion_matrix = np.load("hq_calib_2000_1500_mes/distortion_coefficients.npy")

        self.camera_matrix = np.load("hq_calib_4000_1800/calibration_matrix.npy")
        self.distortion_matrix = np.load("hq_calib_4000_1800/distortion_coefficients.npy")

        # self.camera_matrix = np.load("sony_npy/calibration_matrix.npy")
        # self.distortion_matrix = np.load("sony_npy/distortion_coefficients.npy")

        # init marker parameters
        self.initizer = Init(unit_size=unit_size,marker_size=marker_size)

        self.aruco_dict_type = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        self.marker_size = self.initizer.getMarkerSize()
        self.marker_edge = self.initizer.getMarkerEdge()
        self.offset_matrix = self.initizer.getMatrixOffset()
        self.aruco_parameters = self.initizer.getArucoParameters()

        # create visualization
        self.vis = Visaliser(self.camera_matrix, self.distortion_matrix, vis_enable)

        self.xyz_pos = []
        self.angles = []
        self.trans_mat_pos = []

        self.last_x = 0
        self.last_y = 0

#def drawWithRombiCube(self):


    def getDrawing(self):
        return self.xyz_pos

    def getAngles(self):
        return self.angles

    def undistort(self, frame):
        h, w = frame.shape[:2]
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix,self.distortion_matrix, (w,h), 1, (w,h))
        dst = cv2.undistort(frame,self.camera_matrix, self.distortion_matrix, None, self.newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h,x:x+w]
        return dst

    def xyRombiCenter(self, list_x, list_y):
        self.last_x = sum(list_x)/len(list_x)
        self.last_y = sum(list_y)/len(list_y)

    def detectMarkers(self, img):
        WINDOW = 200
        h, w = img.shape[:2]
        if self.last_x != 0 or self.last_y != 0:    # chceck if last val is valid (0 means init val)
            if self.last_y - WINDOW > 0 and self.last_x - WINDOW > 0:     #check if WINDOW doesn overlap pic (left, top)
                if self.last_y + WINDOW < h and self.last_x + WINDOW < w:     #check if WINDOW doesn overlap pic (right, bott)
                    crop_img = img[int(self.last_y - WINDOW):int(self.last_y + WINDOW),int(self.last_x - WINDOW):int(self.last_x + WINDOW)]
                
                else: 
                    crop_img = img[int(self.last_y - WINDOW):int(h), int(self.last_x- WINDOW):int(w)]
                
                corners, index_of_marker, rejected_img_points = cv2.aruco.detectMarkers(
                        crop_img, self.aruco_dict_type, parameters=self.aruco_parameters)
                    
                if len(corners) > 0:
                    for i in range(len(index_of_marker)):
                        for j in range(4):
                            corners[i][0][j][0] = corners[i][0][j][0] + (self.last_x - WINDOW)
                            corners[i][0][j][1] = corners[i][0][j][1] + (self.last_y - WINDOW)
                    return corners, index_of_marker
            
            else:
                crop_img = img[0:int(self.last_y + WINDOW), 0:int(self.last_x + WINDOW)]
                corners, index_of_marker, rejected_img_points = cv2.aruco.detectMarkers(
                        crop_img, self.aruco_dict_type, parameters=self.aruco_parameters)
                return corners, index_of_marker
            # now i have croped image

        corners, index_of_marker, rejected_img_points = cv2.aruco.detectMarkers(
                img, self.aruco_dict_type, parameters=self.aruco_parameters)
        return corners, index_of_marker

    def estimatePose(self, frame):
        self.transform_matrix_array = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # time2 = time.time()
        # frame = self.undistort(frame)
        # print("=CYCLE=: {0}".format(time.time()-time2))

        # detect markers
        time1 = time.time()
        corners, index_of_marker = self.detectMarkers(gray)
        print("search: {0}".format(time.time()-time1))
        print("succes: {0}".format(len(corners)))
        # If markers are detected

        if len(corners) > 0:
            rombi_center_x = []
            rombi_center_y = []
            time1 = time.time()
            for i in range(0, len(index_of_marker)):

                success, rvec, tvec, d = cv2.solvePnPGeneric(self.marker_edge,
                                                             corners[i],
                                                            #  cameraMatrix=np.zeros((3,3)),
                                                            #  cameraMatrix=self.newcameramtx,
                                                            #  distCoeffs=np.zeros((1,5)),
                                                             self.camera_matrix,
                                                             self.distortion_matrix,
                                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)

                # cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw Axis

                # for j in range(success):
                #     self.vis.showAxis(frame, rvec[j], tvec[j])

                # ======== ZRYCHLOVACI ULTRA MEGA GIGA ALGORITMUS =======
                for j in range(4):
                    rombi_center_x.append(corners[i][0][j][0])
                    rombi_center_y.append(corners[i][0][j][1])
                # =======================================================
                self.createTransfromMatrixArray(
                    rvec=rvec, tvec=tvec, index_of_marker=index_of_marker[i], succes=success)
            print("estimate: {0}".format(time.time()-time1))

            # ======== ZRYCHLOVACI ULTRA ALGORITMUS =======
            self.xyRombiCenter(rombi_center_x,rombi_center_y)
            # =============================================

            time1 = time.time()
            transformation_center = self.centerMarkers()
            print("center: {0}".format(time.time()-time1))

            if len(transformation_center)>1:
                time1 = time.time()
                transformation_selected, good_rotation_count = self.removeBadCandidates(transformation_center)
                print("filter: {0}".format(time.time()-time1))

                # for trans_mat in transformation_selected:
                #     self.vis.show3D(trans_matrix=trans_mat, auto_clear=False)
                # self.vis.clear3D()

                time1 = time.time()
                for trans_mat in transformation_selected:
                    self.vis.showAxis2(frame,  trans_mat, 0.015)
                print("show axis?: {0}".format(time.time()-time1))

                if good_rotation_count > 0:
                    time1 = time.time()
                    self.transformation_finall_center = trans.fuseArucoRotation(transformation_selected,corners, index_of_marker)
                    self.transformation_finall_tip = trans.tipPosition(self.transformation_finall_center)
                    # self.transformation_finall_tip = self.transformation_finall_center
                    self.vis.showAxis2(frame,  self.transformation_finall_tip, 0.01)
                    print("transform tip: {0}".format(time.time()-time1))

                    time1 = time.time()
                    self.xyz_pos.append([self.transformation_finall_tip[0,3],self.transformation_finall_tip[1,3],self.transformation_finall_tip[2,3]])
                    self.angles.append(trans.getAngles(self.transformation_finall_tip))
                    self.trans_mat_pos.append(self.transformation_finall_center)
                    self.vis.show3D(trans_matrix=trans_mat, auto_clear=True)
                    print("colect data: {0}".format(time.time()-time1))
        

        # time1 = time.time()
        # frame = cv2.flip(frame, 1)
        # print("flip: {0}".format(time.time()-time1))

    def createTransfromMatrixArray(self, rvec, tvec, index_of_marker, succes):

        for i in range(succes):

            tvec_temp = [tvec[i][0][0], tvec[i][1][0], tvec[i][2][0]]
            rvec_temp = rvec[i]

            self.transform_matrix_array.append(
                [index_of_marker, trans.rvecTvecToTransfMatrix(tvec=tvec_temp, rvec=rvec_temp)])

    def centerMarkers(self):
        transform_array = []
        transform_array = trans.centerAruco2(
            self.transform_matrix_array, self.offset_matrix)
        return transform_array
    def centerMarkers2(self):
        transform_array = []
        transform_array = trans.centerAruco3(
            self.transform_matrix_array, self.offset_matrix)
        return transform_array

    def removeBadCandidates(self, transformation_center):
        best_rotation = []
        best_translation = []

        # test quaternion
        quat_array = self.getQuaternion(transformation_center)
        similar_quat_index = self.getSimilarQuat(quat_array)
        best_rotation = self.getBestRot(transformation_center, similar_quat_index)
        # if len(transformation_center) > 0:
        #     best_rotation = self.removeBadCandidates_rotation(transformation_center)
        if len(best_rotation) > 0:
            best_translation = self.removeBadCandidates_translacion(best_rotation)
        return best_translation, len(best_translation)

    def getQuaternion(self, transformation_array):
        quat_array = []
        for trans_mat in transformation_array:
            quat_array.append(trans.getQuaternion(trans_matrix=trans_mat))
        return quat_array
    def getSimilarQuat(self, quat_array):
        okay_quat = []

        for i, quat1 in enumerate(quat_array):
            okay_quat_temp = []
            for j, quat2 in enumerate(quat_array):
                are_similar = trans.quaternionAreAlmostSame(quat1,quat2)
                if are_similar:
                    if j not in okay_quat_temp:
                        okay_quat_temp.append(j)
                #print(i, " with ", j, " are ", are_similar)
            okay_quat.append(okay_quat_temp)

        selected_quat_array_index = 0
        for new_index, list in enumerate(okay_quat):
            if len(list) > len(okay_quat[selected_quat_array_index]):
                selected_quat_array_index = new_index
        return okay_quat[selected_quat_array_index]

    def getBestRot(self, trans_mat, index):
        return_trans_mat = []
        for i in index:
            return_trans_mat.append(trans_mat[i])

        return return_trans_mat
    def removeBadCandidates_rotation(self, transformation_array):
        def getAxisAngle(transformation_array):
            rvec_long_array = []
            for trans_mat in transformation_array:
                rot_mat = trans.rot.from_matrix(trans_mat[:3, :3])
                rot_vec = trans.rot.as_rotvec(rot_mat)
                theta1 = np.sqrt(
                    np.power(rot_vec[0], 2) + np.power(rot_vec[1], 2) + np.power(rot_vec[2], 2))
                rvec_1_long = [rot_vec[0]/theta1,
                               rot_vec[1]/theta1, rot_vec[2]/theta1, theta1]
                rvec_long_array.append(rvec_1_long)
            return rvec_long_array




        def removeBadAxisAngle(rvec_long_array, threshold=0.25):
            rvec_best_match_array = []
            rvec_each_to_each = []
            for rvec_current in rvec_long_array:
                rvec_best_match_array.append(rvec_each_to_each.copy())
                rvec_each_to_each = []
                for rvec_for_compare in rvec_long_array:
                    rvec_potentional = []
                    for i in range(0, len(rvec_for_compare)-1):
                        temp = np.abs(rvec_current[i]-rvec_for_compare[i])
                        if temp < threshold:
                            rvec_potentional.append(temp)

                    if len(rvec_potentional) == 3:
                        rvec_each_to_each.append(rvec_for_compare)

            selected_rvec_array_index = 0
            for new_index, list in enumerate(rvec_best_match_array):
                if len(list) > len(rvec_best_match_array[selected_rvec_array_index]):
                    selected_rvec_array_index = new_index
            return rvec_best_match_array[selected_rvec_array_index]

        def selectBestTransformation(transformation_array, list_of_best_rvec, rvec_long_array):
            index_array_of_trans_mat_to_use = []
            for k in list_of_best_rvec:
                index_array_of_trans_mat_to_use.append(
                    rvec_long_array.index(k))

            trans_matrix_to_use = []
            for index in index_array_of_trans_mat_to_use:
                trans_matrix_to_use.append(transformation_array[index])
            return trans_matrix_to_use

        rvec_axis_angle = getAxisAngle(transformation_array)
        list_of_best_rvec = removeBadAxisAngle(rvec_axis_angle, threshold=0.2)
        trans_matrix_to_use = selectBestTransformation(
            transformation_array, list_of_best_rvec, rvec_axis_angle)
        return trans_matrix_to_use

    def removeBadCandidates_translacion(self, transformation_array):
        best_translation, centers_R3, good_indices = trans.removeBadCandidates(np.array(transformation_array), self.camera_matrix, self.distortion_matrix)
        return best_translation
