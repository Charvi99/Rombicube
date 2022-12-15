import cv2
from RombiCube import RombiCube

import numpy as np
import matplotlib.pyplot as plt
import server2 as server

if __name__ == '__main__':

    
    #capture = cv2.VideoCapture(0)
    
    my_server = server.ImageServer()
    rombiCube = RombiCube(unit_size=0.05, marker_size=0.015)

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
        try:
            frame = my_server.getFrame()
            rombiCube.estimatePose(frame=frame)
            #rombiCube.drawWithRombiCube()
            
            frame_prew = cv2.resize(frame,[820,616])
            cv2.imshow('Estimated Pose', frame_prew)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            

        except:
            pass    
        
        

    
    
    # Creating dataset

    
    # Creating figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    # Creating plot
    drawing = rombiCube.getDrawing()
    arr = np.array(drawing)
    np.savetxt("foo2.csv", arr, delimiter=",")
    ax.plot(arr[:,0],arr[:,1],arr[:,2], color = "green")
    plt.title("simple 3D scatter plot")
    
    ax.set_xlim3d([-1, 0])
    ax.set_ylim3d([-1, 0])
    ax.set_zlim3d([4, 5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
        
    # show plot
    plt.show()

    #capture.release()
    cv2.destroyAllWindows()
