
import numpy as np
import cv2
from cv2 import aruco
import pickle
import glob
import time
import sys
import select


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 9 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)

aurocParameters = cv2.aruco.DetectorParameters_create()
aurocParameters.adaptiveThreshWinSizeMin = 3
aurocParameters.adaptiveThreshWinSizeMax = 23
aurocParameters.adaptiveThreshWinSizeStep = 5
aurocParameters.cornerRefinementMaxIterations = 50
aurocParameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aurocParameters.cornerRefinementMinAccuracy = 0.001
aurocParameters.cornerRefinementWinSize = 10
aurocParameters.maxMarkerPerimeterRate  = 4
aurocParameters.polygonalApproxAccuracyRate  = 0.01
aurocParameters.minMarkerPerimeterRate  = 0.03
# aurocParameters.minSideLengthCanonicalImg  = 1
aurocParameters.useAruco3Detection = True
# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.028,
        markerLength=0.022,
        dictionary=ARUCO_DICT)

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime


# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
#images = glob.glob('./camera-pic-of-charucoboard-*.jpg')

#cap = cv2.VideoCapture(0)

images = []
i = 0

capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 4032
height = 3040    
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


while True:
    
    ret, frame = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=ARUCO_DICT,parameters=aurocParameters)  
    print("aruco count:")    
    print(len(corners))
    if len(corners) > 3:
    # images.append(frame.astype(np.uint8))
        gray = aruco.drawDetectedMarkers(
                image=gray, 
                corners=corners)
        
        
        try:
        # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=CHARUCO_BOARD)
        except:
            response = 0;
            print("Error")
            
        print("response:")   
        print(response)
            # Grayscale the image
        
        # Find aruco markers in the query image
        # corners, ids, _ = aruco.detectMarkers(
        #         image=gray,
        #         dictionary=ARUCO_DICT)

        # Outline the aruco markers found in our query image


        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        if response > 40:
            # Add these corners and ids to our calibration arrays
            corners_all.append(charuco_corners)
            ids_all.append(charuco_ids)
            
            # Draw the Charuco board we've detected to show our calibrator the board was properly detected
            gray = aruco.drawDetectedCornersCharuco(
                    image=gray,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids)
        
            # If our image size is unknown, set it now
            if not image_size:
                image_size = gray.shape[::-1]
        
            # Reproportion the image, maxing width or height at 1000
            # Pause to display each image, waiting for key press
            # cv2.imshow('Charuco board', img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            print("already have:") 
            print(len(corners_all))

        
        
        
    proportion = max(gray.shape) / 1000.0
    gray = cv2.resize(gray, (int(gray.shape[1]/proportion), int(gray.shape[0]/proportion)))
        
    cv2.imshow('frame',gray)
    time.sleep(0.3) 
    print("==========:") 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break 
    
    if len(corners_all) == 200:
        break

calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
# Print matrix and distortion coefficient to the console
print(cameraMatrix)
print(distCoeffs)

np.save("calibration_matrix", cameraMatrix)
np.save("distortion_coefficients", distCoeffs)
   

# ==================================================
# ==================================================

# # Destroy any open CV windows
# # cv2.destroyAllWindows()

# # Make sure at least one image was found
# if len(images) < 1:
#     # Calibration failed because there were no images, warn the user
#     print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
#     # Exit for failure
#     exit()

# # Make sure we were able to calibrate on at least one charucoboard by checking
# # if we ever determined the image size
# if not image_size:
#     # Calibration failed because we didn't see any charucoboards of the PatternSize used
#     print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
#     # Exit for failure
#     exit()

# # Now that we've seen all of our images, perform the camera calibration
# # based on the set of points we've discovered
# calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
#         charucoCorners=corners_all,
#         charucoIds=ids_all,
#         board=CHARUCO_BOARD,
#         imageSize=image_size,
#         cameraMatrix=None,
#         distCoeffs=None)
    
# # Print matrix and distortion coefficient to the console
# print(cameraMatrix)
# print(distCoeffs)

# np.save("calibration_matrix", cameraMatrix)
# np.save("distortion_coefficients", distCoeffs)
    
# # Save values to be used where matrix+dist is required, for instance for posture estimation
# # I save files in a pickle file, but you can use yaml or whatever works for you
# f = open('calibration.pckl', 'wb')
# pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
# f.close()
    
# # Print to console our success
# print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))