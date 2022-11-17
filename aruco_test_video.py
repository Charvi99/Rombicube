import imp
from multiprocessing.connection import wait
from re import T
import numpy as np
import cv2
import sys

from pyparsing import null_debug_action
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import matplotlib.pyplot as plt
import visualisation as vis
import transform as trans
import start_init as init
import time
from datetime import timedelta
import dodecaPen
import pyrealsense2 as rs

x = 0
y = 0
z = 0
trans_matrix = np.eye(4)


################### INFRARED D455 ###################
# pipeline = rs.pipeline()

# config = rs.config()
# config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
# config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

# colorizer = rs.colorizer()
# pipeline.start(config)

# profile = pipeline.get_active_profile()
# infrared_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 2))
# infrared_intrinsics = infrared_profile.get_intrinsics()

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, ax):

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
	parameters = cv2.aruco.DetectorParameters_create()
	parameters.adaptiveThreshWinSizeMin = 5
	parameters.adaptiveThreshWinSizeMax = 5
	parameters.adaptiveThreshWinSizeStep = 100
	parameters.cornerRefinementWinSize = 10
	parameters.cornerRefinementMethod = 1
	parameters.cornerRefinementMinAccuracy = 0.001
	parameters.cornerRefinementMinAccuracy = 0.001
	parameters.cornerRefinementMaxIterations = cv2.aruco.CORNER_REFINE_CONTOUR

	start = time.time()
	corners, idx_g, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)
	end = time.time()

	# print("Time:")
	# print(timedelta(seconds=end-start))
	# time.sleep(0.5)
	trans_matrix_array_1 = []
	trans_matrix_array_2 = []

	transform_array_1 = []
	transform_array_2 = []
	mar_lenght = init.marker_size()
	aruco_edges = np.array([[-mar_lenght/2, mar_lenght/2,	0],
                         [mar_lenght/2, mar_lenght/2,	0],
                         [mar_lenght/2,	-mar_lenght/2,	0],
                         [-mar_lenght/2,	-mar_lenght/2,	0]], dtype='float32').reshape((4, 1, 3))
	# If markers are detected
	if len(corners) > 0:

		for i in range(0, len(idx_g)):

			success, rvec, tvec, d = cv2.solvePnPGeneric(
				aruco_edges, corners[i], matrix_coefficients, distortion_coefficients, flags=cv2.SOLVEPNP_IPPE_SQUARE)

			cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
			for j in range(success):
				cv2.drawFrameAxes(frame, matrix_coefficients,
				                  distortion_coefficients, rvec[j], tvec[j], 0.01)

			tvec_1 = [tvec[0][0][0], tvec[0][1][0], tvec[0][2][0]]
			tvec_2 = [tvec[1][0][0], tvec[1][1][0], tvec[1][2][0]]
			rvec_1 = rvec[0]
			rvec_2 = rvec[1]

			trans_matrix_array_1.append(
				[idx_g[i], trans.rvecTvecToTransfMatrix(tvec=tvec_1, rvec=rvec[0])])
			trans_matrix_array_2.append(
				[idx_g[i], trans.rvecTvecToTransfMatrix(tvec=tvec_2, rvec=rvec[1])])

		transform_array_1 = trans.centerAruco2(trans_matrix_array_1)
		transform_array_2 = trans.centerAruco2(trans_matrix_array_2)

		transform_array = []
		transform_array.extend(transform_array_1)
		transform_array.extend(transform_array_2)

		rvec_long_array = []
		rvec_best_match_array = []
		for trans_mat in transform_array:
			rot_mat = trans.rot.from_matrix(trans_mat[:3, :3])
			rot_vec = trans.rot.as_rotvec(rot_mat)
			theta1 = np.sqrt(
				np.power(rot_vec[0], 2) + np.power(rot_vec[1], 2) + np.power(rot_vec[2], 2))
			rvec_1_long = [rot_vec[0]/theta1,
                            rot_vec[1]/theta1, rot_vec[2]/theta1, theta1]
			rvec_long_array.append(rvec_1_long)
			#print(rvec_1_long)

		rvec_each_to_each = []
		for rvec_current in rvec_long_array:
			rvec_best_match_array.append(rvec_each_to_each.copy())
			rvec_each_to_each = []
			for rvec_for_compare in rvec_long_array:
				print("[")
				rvec_potentional = []
				for i in range(0, len(rvec_for_compare)):
					temp = np.abs(rvec_current[i]-rvec_for_compare[i])
					if temp < 0.3:
						rvec_potentional.append(temp)
					print(temp)
				print("]")
				if len(rvec_potentional) == 4:
					rvec_each_to_each.append(rvec_for_compare)

		selected_rvec_array_index = 0
		for new_index, list in enumerate(rvec_best_match_array):
			print(len(list))
			if len(list) > len(rvec_best_match_array[selected_rvec_array_index]):
				selected_rvec_array_index = new_index

		index_array_of_trans_mat_to_use = []
		for k in rvec_best_match_array[selected_rvec_array_index]:
			index_array_of_trans_mat_to_use.append(rvec_long_array.index(k))

		trans_matrix_to_use = []
		for index in index_array_of_trans_mat_to_use:
			trans_matrix_to_use.append(transform_array[index])

		# rot_dif = []
		# reference_transform = transform_array[0]

		# for i in range(1,len(transform_array)):
		# 	comparision = trans.compareRotationInTransMatrix(reference_transform,transform_array[i])
		# 	print(i, ". ", comparision)
		# 	rot_dif.append(comparision)
		good_centered_aruco, centers_R3, good_indices = trans.removeBadCandidates(
			np.array(trans_matrix_to_use), matrix_coefficients, distortion_coefficients)

		for trans_mat in good_centered_aruco:
			vis.draw3Dscatter(ax, trans_matrix=trans_mat)
		# plt.pause(1)
		#vis.draw3Dscatter(ax,trans_matrix=transform_array_2[0])

		ax.cla()

	frame = cv2.flip(frame, 1)

	return frame,  [transform_array_1, transform_array_2], idx_g, corners


if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-k", "--K_Matrix", required=True,
	                help="Path to calibration matrix (numpy file)")
	ap.add_argument("-d", "--D_Coeff", required=True,
	                help="Path to distortion coefficients (numpy file)")
	ap.add_argument("-t", "--type", type=str,
	                default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
	args = vars(ap.parse_args())

	#======================== video estimation ======================

	if ARUCO_DICT.get(args["type"], None) is None:
		print(f"ArUCo tag type '{args['type']}' is not supported")
		sys.exit(0)

	aruco_dict_type = ARUCO_DICT[args["type"]]
	calibration_matrix_path = args["K_Matrix"]
	distortion_coefficients_path = args["D_Coeff"]

	k = np.load("webcam_npy/calibration_matrix.npy")
	d = np.load("webcam_npy/distortion_coefficients.npy")

	#video = cv2.VideoCapture(1, cv2.CAP_DSHOW)

	capture = cv2.VideoCapture(0)
	# codec = cv2.VideoWriter_fourcc(	'M', 'J', 'P', 'G'	)
	# capture.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, codec)
	# capture.set(cv2.CAP_PROP_XI_FRAMERATE, 30)
	# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
	# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
	# width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	# height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# print(width,height)
	offset_matrix = init.init_offset_matrix(localization_unit_size=0.036)

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_box_aspect([1, 1, 1])
	# vis.draw3Dscatter(ax,np.eye(4,4))
	# vis.draw3Dscatter_origin_offset(ax, offset_matrix)

	#######################
	##### FOR DEDECA ######
	#######################
	# params = dodecaPen.parameters()  ## initializes the parameters
	# data = dodecaPen.txt_data()
	#######################

	while True:

		ret, frame = capture.read()
		################### INFRARED D455 ###################

		# frames = pipeline.wait_for_frames()

		# infrared_frame_zero = frames.get_infrared_frame(1)
		# infrared_frame_one  = frames.get_infrared_frame(2)
		# image = np.asanyarray(infrared_frame_zero.get_data())
		if not ret:
			# if not True:
			break
		#######################
		##### FOR DEDECA ######
		#######################
		# frame_dodeca_without = copy.copy(frame)
		# frame_dodeca_ape = copy.copy(frame)
		# frame_dodeca_dpr = copy.copy(frame)
		# frame_my= copy.copy(frame)

		# frame_gray_draw,pose_without_opt, pose_APE,pose_DPR,visib_flag = dodecaPen.find_pose(frame,params,data)
		# print( np.array(pose_DPR[0][0:3]))
		# print( np.array(pose_DPR[0][3:6]))
		# cv2.drawFrameAxes(frame_dodeca_without, k, d, np.array(pose_without_opt[0][0:3]), np.array(pose_without_opt[0][3:6]), 0.01)
		# cv2.drawFrameAxes(frame_dodeca_ape, k, d, np.array(pose_APE[0][0:3]), np.array(pose_APE[0][3:6]), 0.01)
		# cv2.drawFrameAxes(frame_dodeca_dpr, k, d, np.array(pose_DPR[0][0:3]), np.array(pose_DPR[0][3:6]), 0.02)
		# final_rotation = trans.rvecsTvecsToTransfMatrix2(np.array(pose_DPR[0][0:3]), np.array(pose_DPR[0][3:6]))
		# vis.draw3Dscatter(ax,final_rotation)

		#######################

		output_frame, output_transMatrixArray, output_ids, output_corners = pose_esitmation(
			frame, aruco_dict_type, k, d, ax)

		# if output_ids_transMatrixArray[0] is not None:
		# 	if len(output_ids_transMatrixArray[0]) > 0:
		# 		vis.draw3Dscatter(ax,output_ids_transMatrixArray[1][0])

		if False:

			if output_transMatrixArray is not None and output_ids is not None:
				if len(output_ids) > 0:

					centered_aruco = trans.centerAruco(
						output_transMatrixArray, output_ids, offset_matrix)

					good_centered_aruco, centers_R3, good_indices = trans.removeBadCandidates(
						np.array(centered_aruco), k, d)
					final_rotation = trans.fuseArucoRotation(
						good_centered_aruco, output_ids, output_corners)

					#vis.draw3Dscatter_origin_offset(ax,good_centered_aruco)
					#vis.draw3Dscatter(ax,final_rotation)

					# final_rotation = np.eye(4,4)
					# for current_rotatin in good_centered_aruco:
					# 	final_rotation[:3, :3] = np.matmul(final_rotation[:3, :3],current_rotatin[:3, :3])
					# 	final_rotation[0,3] = current_rotatin[0,3]
					# 	final_rotation[1,3] = current_rotatin[1,3]
					# 	final_rotation[2,3] = current_rotatin[2,3]
					vis.draw3Dscatter(ax, final_rotation)
			ax.cla()

		#print(output_ids_transMatrixArray[0])
		cv2.imshow('Estimated Pose', output_frame)
		# cv2.imshow('dodeca Pose', frame_dodeca_without)
		# cv2.imshow('dodeca APE Pose', frame_dodeca_ape)
		# cv2.imshow('dodeca DPR Pose', frame_dodeca_dpr)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	capture.release()
	cv2.destroyAllWindows()
