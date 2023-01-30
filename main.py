import cv2
from RombiCube import RombiCube
from unity_socket import send

import numpy as np
import matplotlib.pyplot as plt
import server2 as server
import time

# from picamera2 import Picamera2, Preview
import sys
import select
# import tty
# import termios

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

# old_settings = termios.tcgetattr(sys.stdin)

if __name__ == '__main__':


    # picam2 = Picamera2()
    # camera_config = picam2.create_still_configuration(buffer_count=2)
    
    # camera_config2 = picam2.create_preview_configuration()

    # We're going to set up some configuration structures, apply each one in
    # turn and see if it gave us the configuration we expected.

    res_1 = (2000,1500)
    res_2 = (3000,2000)
    res_3 = (4000,3000)
    res_4 = (4000,1800)

    # picam2.preview_configuration.size = res_4
    # picam2.preview_configuration.format = "BGR888"
    # picam2.preview_configuration.controls.ExposureTime = 10000
    # picam2.configure("preview")


    # picam2.start_preview(Preview.QT)
    # picam2.start()
    # time.sleep(1)
    
    #capture = cv2.VideoCapture(0)
    
    #my_server = server.ImageServer()
    
    rombiCube = RombiCube(unit_size=0.05, marker_size=0.015,vis_enable=True)

    # tty.setcbreak(sys.stdin.fileno())

    while True:

        #ret, frame = capture.read()
        # if not ret:
        #     # if not True:
        #     break
        #rombiCube.estimatePose(frame=frame)
        # cv2.imshow('Estimated Pose', frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
        #try:
        
            # frame = picam2.capture_array("main")
            frame = np.zeros((4000,3000,3),dtype=np.uint8)
            time1 = time.time()
            rombiCube.estimatePose(frame=frame)
            [x,y,z] = rombiCube.getXYZ()
            [qx,qy,qz,w] = rombiCube.getQuat()
            
            send(x,y,z,qx,qy,qz,w)
            print("cycle: {0}".format(time.time()-time1))
            print("=============================")

            # if isData():
            #     c = sys.stdin.read(1)
            #     break

            # plt.figure(1)
            # plt.imshow(dst) 
            # plt.pause(0.1)
              
        
    # termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)    
    
    
    # Creating dataset

    
    # Creating figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    # Creating plot
    drawing = rombiCube.getDrawing()
    arr = np.array(drawing)
    np.savetxt("6dof_xyz.csv", arr, delimiter=",")
    # dof_mes_array = np.array(rombiCube.trans_mat_pos)
    # print(dof_mes_array)
    # np.savetxt("6dof_mes.csv", dof_mes_array, delimiter=",")
    ax.plot(arr[:,0],arr[:,1],arr[:,2], color = "green")
    plt.title("simple 3D scatter plot")
    
    # ax.set_xlim3d([-180, 180])
    # ax.set_ylim3d([-180, 180])
    # ax.set_zlim3d([-180, 180])

    ax.set_xlim3d([-0.2, 0.2])
    ax.set_ylim3d([-0.2, 0.2])
    ax.set_zlim3d([0.2, 0.6])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
        
    # show plot
    plt.show()

    #capture.release()
    # cv2.destroyAllWindows()
