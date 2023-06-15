import numpy as np
import matplotlib.pyplot as plt
import cv2
import transform as trans

class Visaliser():
    def __init__(self, camera_matrix, distortion_matrix, vis_enable):
        # setings for scatter 
        if vis_enable:
            plt.ion()
            self.fig = plt.figure(0)
            self.ax = self.fig.add_subplot(projection='3d')
            self.ax.set_box_aspect([1, 1, 1])

        # setting for figure
        self.camera_matrix = camera_matrix
        self.distortion_matrix = distortion_matrix
        self.enable = vis_enable



    def setScatter(self):
        if self.enable == False:
            return
        self.ax.set_xlim3d([-1, 0])
        self.ax.set_ylim3d([-1, 0])
        self.ax.set_zlim3d([1, 3])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def show3D(self,trans_matrix, auto_clear = True):
        if self.enable == False:
            return
        self.setScatter()
        self.x = trans_matrix[0,3]
        self.y = trans_matrix[1,3]
        self.z = trans_matrix[2,3]

        for i in range(3):
            self.drawSingleAxis(trans_matrix,i)
        
        plt.draw()
        plt.pause(0.001)
        if auto_clear:
            self.ax.cla()

    def drawSingleAxis(self, trans_matrix=np.eye(4,4), axis=0, lenght=0.2, line_width=3):
        if self.enable == False:
            return
        axis_matrix = np.eye(4,4)
        axis_matrix[axis,3] = lenght
        trans_axis_matrix = np.matmul(trans_matrix,axis_matrix)
        x_new = trans_axis_matrix[0,3]
        y_new = trans_axis_matrix[1,3]
        z_new = trans_axis_matrix[2,3]
        self.ax.plot(
            np.array([self.x, x_new]),  ###
            np.array([self.y, y_new]),  ###
            np.array([self.z, z_new]),
            'red' if axis == 0 else 'green' if axis == 1 else 'blue',
            linewidth=line_width) 

    def showAxis(self, frame, rvec, tvec, lenght= 0.01, thickness=1):
        if self.enable == False:
            return        
        cv2.drawFrameAxes(frame, self.camera_matrix,
                                      self.distortion_matrix, rvec=rvec,
                                      tvec=tvec, length=lenght,thickness=thickness)

    def showAxisFromTransMat(self, frame, trans_mat, lenght= 0.01,thickness=1):
        if self.enable == False:
            return
        tvec, rvec = trans.transMatrixToTvecRvec(trans_mat)
        
        cv2.drawFrameAxes(frame, self.camera_matrix,
                                      self.distortion_matrix, rvec=rvec,
                                      tvec=tvec, length=lenght,thickness=thickness)

    def clear3D(self):
        if self.enable == False:
            return
        self.ax.cla()

def draw3Dscatter(self,ax,trans_matrix):
    if self.enable == False:
        return
    #print(trans_matrix)
    x = trans_matrix[0,3]
    y = trans_matrix[1,3]
    z = trans_matrix[2,3]

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([0, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    #######################################################
    eyes = np.eye(4,4)
    eyes[0, 3] = 0.2
    eyes[1, 3] = 0
    eyes[2, 3] = 0
    #print(eyes)
    new_pos = np.matmul(trans_matrix,eyes)
    #print(new_pos)

    x2 = new_pos[0,3]
    y2 = new_pos[1,3]
    z2 = new_pos[2,3]

    ax.plot( np.array([x,x2]),    ###
                np.array([y,y2]),    ###
                np.array([z,z2]),'red',linewidth=4)    ###
    ##########################################################
    eyes = np.eye(4,4)
    eyes[0, 3] = 0
    eyes[1, 3] = 0.2
    eyes[2, 3] = 0
    #print(eyes)
    new_pos = np.matmul(trans_matrix,eyes)
    #print(new_pos)

    x2 = new_pos[0,3]
    y2 = new_pos[1,3]
    z2 = new_pos[2,3]

    ax.plot( np.array([x,x2]),    ###
                np.array([y,y2]),    ###
                np.array([z,z2]),'green',linewidth=4)    ###
    ##########################################################
    eyes = np.eye(4,4)
    eyes[0, 3] = 0
    eyes[1, 3] = 0
    eyes[2, 3] = 0.2
    #print(eyes)
    new_pos = np.matmul(trans_matrix,eyes)
    #print(new_pos)

    x2 = new_pos[0,3]
    y2 = new_pos[1,3]
    z2 = new_pos[2,3]

    ax.plot( np.array([x,x2]),    ###
                np.array([y,y2]),    ###
                np.array([z,z2]),'blue',linewidth=4)    ###


    plt.draw()
    plt.pause(0.001)

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

def tvecToPos(tvec):
    # x = np.interp(tvec[0][0][0],[-0.2,-0.05],[0,100])
    # y = np.interp(tvec[0][0][1],[-0.15,-0.015],[0,80])
    # z = np.interp(tvec[0][0][2],[0.1,1],[0,50])
    x = tvec[0][0][0]
    y = tvec[0][0][1]
    z =tvec[0][0][2]
    return [x, y, z]


def draw3Dscatter_origin_offset(self, ax, offset_array):
    if self.enable == False:
            return
    for offset in offset_array:

        x = offset[0,3]
        y = offset[1,3]
        z = offset[2,3]
        #print(x,y,z)


        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([0, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


        # ax.scatter(	x,    ###
        #             y,    ###
        #             z)    ###

        #######################################################
        x_axis = np.eye(4,4)
        x_axis[0, 3] = 0.2
        x_axis[1, 3] = 0
        x_axis[2, 3] = 0
        #print(eyes)
        new_pos = np.matmul(offset,x_axis)
        #print(new_pos)

        x2 = new_pos[0,3]
        y2 = new_pos[1,3]
        z2 = new_pos[2,3]

        ax.plot( np.array([x,x2]),    ###
                    np.array([y,y2]),    ###
                    np.array([z,z2]),'red')    ###
        ##########################################################
        y_axis = np.eye(4,4)
        y_axis[0, 3] = 0
        y_axis[1, 3] = 0.2
        y_axis[2, 3] = 0
        #print(eyes)
        new_pos = np.matmul(offset,y_axis)
        #print(new_pos)

        x2 = new_pos[0,3]
        y2 = new_pos[1,3]
        z2 = new_pos[2,3]

        ax.plot( np.array([x,x2]),    ###
                    np.array([y,y2]),    ###
                    np.array([z,z2]),'green')    ###
        ##########################################################
        z_axis = np.eye(4,4)
        z_axis[0, 3] = 0
        z_axis[1, 3] = 0
        z_axis[2, 3] = 0.2
        #print(eyes)
        new_pos = np.matmul(offset,z_axis)
        #print(new_pos)

        x2 = new_pos[0,3]
        y2 = new_pos[1,3]
        z2 = new_pos[2,3]

        ax.plot( np.array([x,x2]),    ###
                    np.array([y,y2]),    ###
                    np.array([z,z2]),'blue')    ###


        plt.draw()
    plt.pause(0.001)
    #plt.pause(0.01)
