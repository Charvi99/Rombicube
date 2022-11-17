'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


from multiprocessing.connection import wait
from re import T
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT, aruco_display
import argparse
import time


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

	'''
	frame - Frame from the video stream
	matrix_coefficients - Intrinsic matrix of the calibrated camera
	distortion_coefficients - Distortion coefficients associated with your camera

	return:-
	frame - The frame with the axis drawn on it
	'''

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
	parameters = cv2.aruco.DetectorParameters_create()


	corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters)
	
        # If markers are detected
	if len(corners) > 0: 
		
		for i in range(0, len(ids)):
			print("Index: {}" .format(ids[i]))
			print("Corner:")
			for j in range(0, 4):
				print("({}): {}".format(j, corners[i][0][j]))
				r = np.interp(1000/(i+1),[1,1000],[0,255])
				g = np.interp(1000/((i+1)*2),[1,1000],[0,255])
				b = np.interp(1000/((i+1)*3),[1,1000],[0,255])
				cv2.putText(img=frame,text=str(j), org = (int(corners[i][0][j][0]),int(corners[i][0][j][1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 0, 0),thickness=2)
			cv2.putText(img=frame,text=str(i), org = (int(corners[i][0][0][0]),int(corners[i][0][0][1])+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),thickness=3)

            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
			rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
																		distortion_coefficients)
			print("==== ORIGINAL TVEC ====")
			print(tvec)
			tvec_x = tvec[0][0][0]
			tvec_y = tvec[0][0][1]
			tvec_z = tvec[0][0][2]

			# tvec_x *= 2
			# tvec_y *= 1
			# tvec_z *= 1

			# print("tvec_x = ", tvec_x)
			# print("tvec_y = ", tvec_y)
			# print("tvec_z = ", tvec_z)

			tvec[0][0][0] = tvec_x
			tvec[0][0][1] = tvec_y
			tvec[0][0][2] = tvec_z

			# print("==== NEW TVEC ====")
			# print(tvec)

			#print(rvec)
			
			# Draw a square around the markers
			cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
			cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

		rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.02, matrix_coefficients, distortion_coefficients)

		

		x = np.interp(40,[0,80],[-0.131,-0.05])
		y = np.interp(30,[0,60],[-0.0765,-0.015])
		z = np.interp(10,[0,80],[-0.05,-0.13])

		tvec[0][0][0] = x
		tvec[0][0][1] = y
		tvec[0][0][2] = 0.2

		cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.02)  

	return frame

if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
	ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
	ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
	args = vars(ap.parse_args())

    
print("Loading image...")
image = cv2.imread("x4.png")
#image = cv2.imread("ArUco.png")
h,w,_ = image.shape
width=600
height = int(width*(h/w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

calibration_matrix_path = args["K_Matrix"]
distortion_coefficients_path = args["D_Coeff"]
k = np.load(calibration_matrix_path)
d = np.load(distortion_coefficients_path)

d = np.zeros(5)


# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("Detecting '{}' tags....".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

detected_markers = aruco_display(corners, ids, rejected, image)
cv2.imshow("Image", detected_markers)

output = pose_esitmation(image, ARUCO_DICT[args["type"]], k, d)
time.sleep(0.5)
cv2.imshow('Estimated Pose', output)

# # Uncomment to save
# cv2.imwrite("output_sample.png",detected_markers)

cv2.waitKey(0)
