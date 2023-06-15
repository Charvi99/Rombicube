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
    try:
        s = socket.socket()   
        
        # connect to the server on local computer 
        s.connect(('127.0.0.1', 5255))
        # s.connect(('192.168.0.1', 5255))

                
        #send data in format x,y,z,qx,qy,qz,qw
        s.send((
            str(float(x)) + "," +  
            str(float(y)) + "," +  
            str(float(z)) +
            ";" + 
            str(float(qx)) + "," +
            str(float(qy)) + "," +
            str(float(qz)) + "," +
            str(float(w))).encode())
        s.close()    
    except:
        pass

    
def sendString(data):
    try:
        s = socket.socket()   
        s.connect(('127.0.0.1', 5255))
        a = (data).encode()
        s.send(a)
        s.close()    
    except:
        pass
    