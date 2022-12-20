
import numpy as np
import cv2
from cv2 import aruco
import pickle
import glob
import time
import server2 as server
from picamera2 import Picamera2, Preview
import sys
import select
import tty
import termios

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

old_settings = termios.tcgetattr(sys.stdin)

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 9 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)

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

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(buffer_count=2)
# camera_config2 = picam2.create_preview_configuration()

# We're going to set up some configuration structures, apply each one in
# turn and see if it gave us the configuration we expected.

res_1 = (2000,1500)
res_2 = (3000,2000)
res_3 = (4000,3000)

picam2.preview_configuration.size = res_1
picam2.preview_configuration.format = "BGR888"
picam2.preview_configuration.controls.ExposureTime = 10000
picam2.configure("preview")


picam2.start_preview(Preview.QT) 
picam2.start()
time.sleep(1)

tty.setcbreak(sys.stdin.fileno())

while True:
    
    if isData():
        c = sys.stdin.read(1)
        frame = picam2.capture_array("main")
        # if prev_img is not frame:
        #     prev_img = frame
        images.append(frame.astype(np.uint8))
        
        i = i + 1
        print(i)
        if i == 15:
            break
        #  c = sys.stdin.read(1)
    frame = picam2.capture_array("main")
    # if prev_img is not frame:
    #     prev_img = frame
    images.append(frame.astype(np.uint8))
    
    i = i + 1
    print(i)
    time.sleep(0.1)
    if i == 50:
        break
termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)  

    # cv2.imshow('frame',frame)
    # time.sleep(0.1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Loop through images glob'ed
for index, img in enumerate(images):
   
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find aruco markers in the query image
    corners, ids, _ = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

    # Outline the aruco markers found in our query image
    img = aruco.drawDetectedMarkers(
            image=img, 
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

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares
    if response > 20:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)
        
        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
       
        # If our image size is unknown, set it now
        if not image_size:
            image_size = gray.shape[::-1]
    
        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
        # Pause to display each image, waiting for key press
        # cv2.imshow('Charuco board', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        print("Not able to detect a charuco board in image: {}".format(index))

# Destroy any open CV windows
# cv2.destroyAllWindows()

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
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
    
# Save values to be used where matrix+dist is required, for instance for posture estimation
# I save files in a pickle file, but you can use yaml or whatever works for you
f = open('calibration.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()
    
# Print to console our success
print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))