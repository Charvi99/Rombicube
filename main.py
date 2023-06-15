import cv2
from RombiCube import RombiCube
from unity_socket import send, sendString

import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import select

import serial

if __name__ == '__main__':
    
    arduino = serial.Serial(port='COM13', baudrate=115200, timeout=.01)

    width = 4032
    height = 3040    
    xyz_serial_caputred = []
    quat_serial_caputred = []
    capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # capture2 = cv2.VideoCapture(3)
    # capture = cv2.VideoCapture(1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # width = 3840
    # height = 2160
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
    capture.set(cv2.CAP_PROP_EXPOSURE, -12.0)
    # Turn off auto exposure
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # set exposure time
    capture.set(cv2.CAP_PROP_EXPOSURE, 40)    
    
    print(capture.get(cv2.CAP_PROP_EXPOSURE))
    time.sleep(0.5)
    
    rombiCube = RombiCube(unit_size=0.074, marker_size=0.026,vis_enable=True)
    
    window_name = "Estimated Pose"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(width/5), int(height/5))

    divider = 4
    frame = np.zeros(((int)(width/divider),(int)(height/divider),3),np.uint8)

    while True:

        ret, frame = capture.read()
        if not ret:
            break;
        
        # ret2, frame2 = capture2.read()
        # if not ret2:
        #     break;
        
        time1 = time.time()
        frame, crop_img = rombiCube.estimatePose(frame=frame)
        [x,y,z] = rombiCube.getXYZ()
        [qx,qy,qz,qw] = rombiCube.getQuat()
        # [a,b,c] = rombiCube.getAngles_last()
        # [ax,ay,az] = rombiCube.getAxisAngles_last()
        
        send(x,y,z,qx,qy,qz,qw)
        if len(arduino.readline()) > 1:
            xyz_serial_caputred.append(rombiCube.getXYZ())
            quat_serial_caputred.append(rombiCube.getQuat())
            sendString("BBBBB")
        # send(x,y,z,ax,ay,az,0)
        #sendString(rombiCube.getTransMatString())
        #send(x,y,z,a,b,c,0)
        print("cycle: {0}".format(time.time()-time1))
        print("=============================")
        
  
        cv2.imshow(window_name, frame)
        # cv2.imshow("usb", frame2)
        cv2.imshow("crop", crop_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
                

    
    # Creating figure
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    
    # Creating plot
    # drawing = rombiCube.getDrawing()
    arr_xyz = np.array(xyz_serial_caputred)
    arr_quat = np.array(quat_serial_caputred)
    np.savetxt("xyz.csv", arr_xyz, delimiter=",", fmt='%f')
    np.savetxt("quat.csv", arr_quat, delimiter=",", fmt='%f')
    
    # ax.plot(arr[:,0],arr[:,1],arr[:,2], color = "green")
    # plt.title("simple 3D scatter plot")


    # ax.set_xlim3d([-0.2, 0.2])
    # ax.set_ylim3d([-0.2, 0.2])
    # ax.set_zlim3d([0.2, 0.6])

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
        
    # # show plot
    # plt.show()

capture.release()
cv2.destroyAllWindows()
