import socket
import random
from time import sleep

global x
global y
global z

global x_dir
global y_dir
global z_dir

    
def send(x,y,z,qx,qy,qz,w):
    s = socket.socket()   
    s.connect(('192.168.0.108', 1755))
              
    # connect to the server on local computer 
    s.send((
        str(int(x*100)) + "," +  
        str(int(y*100)) + "," +  
        str(int(z*100)) + "," +
        ";" + 
        str(int(qx*100)) + "," +
        str(int(qy*100)) + "," +
        str(int(qz*100)) + "," +
        str(int(w*100))).encode())
    s.close()

def changeCoordinates(x,y,z,x_dir,y_dir,z_dir):
    (x, x_dir) = changeSingleCoord(x, x_dir)
    (y, y_dir) = changeSingleCoord(y, y_dir)
    (z, z_dir) = changeSingleCoord(z, z_dir)
    
    return (x,y,z,x_dir,y_dir,z_dir)
             
def changeSingleCoord(number, dir):
    if dir:
        number = number + 5
        if number == 50:
            dir = not dir
    else:
        number = number - 5       
        if number == 0:
             dir = not dir
    return (number, dir)
                              
if __name__ == "__main__":
    x = 0
    y = 25
    z = 0
    x_dir = True
    y_dir = True
    z_dir = True

    while True:
        (x,y,z,x_dir,y_dir,z_dir) = changeCoordinates(x,y,z,x_dir,y_dir,z_dir)
        send(x,y,z)
        sleep(0.1)
        
