# server.py
import io
import socket
import struct
from PIL import Image
import time
import numpy as np
import cv2
import threading


n_recs = 0
cnt_streamed_imgs = 0
summary = []
n_measurements = 100
avg_img_len = 0
W, H = 320, 240
    
class ImageServer(object):
    def __init__(self, *args):
        self.host = "192.168.0.108"
        self.port = 5008
        self.n_recs = 0
        self.cnt_streamed_imgs = 0
        self.summary = []
        self.n_measurements = 100
        self.avg_img_len = 0
        self.W, self.H = 2952, 2218
        self.open_cv_image = np.empty([self.W,self.H,3])
        self.startServer()
        time.sleep(0.5)
        run_thred = threading.Thread(target=self.runServer)
        run_thred.start()
        
    def startServer(self):
        self.server_socket = socket.socket()
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        # Accept a single connection and make a file-like object out of it
        self.connection = self.server_socket.accept()[0].makefile('rb')
    
    def runServer(self, vis=False):
        try:
            test = True
            self.init = time.time()
            while test:
                # Read the length of the image as a 32-bit unsigned int.
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

                if not image_len:
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                try:
                    #reading jpeg image
                    image_stream.write(self.connection.read(image_len))
                    image = Image.open(image_stream)
                except:
                    #if reading raw images: yuv or rgb
                    image = Image.frombytes('L', (W, H), image_stream.read())
                # Rewind the stream
                image_stream.seek(0)
                
                load_img = np.array(image)
                self.open_cv_image = load_img.astype(np.uint8)
                self.open_cv_image = self.open_cv_image[:, :, ::-1].copy()
                
                if vis:
                    self.open_cv_image_small = cv2.resize(self.open_cv_image,[820,616])
                    cv2.imshow('Network Image',self.open_cv_image_small)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            
                self.avg_img_len += image_len
                self.elapsed = (time.time() - self.init)
                self.cnt_streamed_imgs += 1
                if self.elapsed > 10 and self.elapsed < 11:
                    #record number of images streamed in about 10secs
                    self.avg_img_len = self.avg_img_len / self.cnt_streamed_imgs
                    print("{} | Nbr_frames: {} - Elapased Time: {:.2f} | Average img length: {:.1f}]".format(self.n_recs, self.cnt_streamed_imgs, self.elapsed, self.avg_img_len) )
                    self.summary.append( [self.cnt_streamed_imgs, self.elapsed, self.avg_img_len] )
                    self.n_recs += 1
                    #reset counters
                    self.init = time.time()
                    self.cnt_streamed_imgs = 0
                    self.avg_img_len = 0
                if self.n_recs == self.n_measurements:
                    #Number of measurements
                    test = False

                #Write summary
            # with open("stream_perf_07.txt", "w") as file:
            #     file.write("nbr_images, elapsed(sec), avg_img_size\n")
            #     for record in summary:
            #         file.write("{}, {}, {}\n".format( record[0], record[1], record[2]))

        finally:
            self.connection.close()
            self.server_socket.close()
    def getFrame(self):
        return self.open_cv_image.astype(np.uint8)


if __name__ == '__main__':
    my_server = ImageServer()
    my_server.startServer()
    
    my_server.runServer(vis=True)

        

# import io
# import socket
# import struct
# from PIL import Image
# import time
# import numpy
# import cv2

# host = '192.168.0.108'

# if __name__ == '__main__':
#     # Start a socket listening for connections on 0.0.0.0:8000 
#     # (0.0.0.0 means all interfaces)
#     server_socket = socket.socket()
#     server_socket.bind((host, 5008))
#     server_socket.listen(1)
#     # Accept a single connection and make a file-like object out of it
#     connection = server_socket.accept()[0].makefile('rb')
#     ##############
#     # Parameters
#     ##############
#     n_recs = 0
#     cnt_streamed_imgs = 0
#     summary = []
#     n_measurements = 100
#     avg_img_len = 0
#     W, H = 320, 240

#     try:
#         test = True
#         init = time.time()
#         while test:
#             # Read the length of the image as a 32-bit unsigned int.
#             image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

#             if not image_len:
#                 break
#             # Construct a stream to hold the image data and read the image
#             # data from the connection
#             image_stream = io.BytesIO()
#             try:
#                 #reading jpeg image
#                 image_stream.write(connection.read(image_len))
#                 image = Image.open(image_stream)
#             except:
#                 #if reading raw images: yuv or rgb
#                 image = Image.frombytes('L', (W, H), image_stream.read())
#             # Rewind the stream
#             image_stream.seek(0)
            
#             open_cv_image = numpy.array(image)
#             open_cv_image = open_cv_image[:, :, ::-1].copy()
#             open_cv_image = cv2.resize(open_cv_image,[820,616])
#             cv2.imshow('Network Image',open_cv_image)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
        
#             avg_img_len += image_len
#             elapsed = (time.time() - init)
#             cnt_streamed_imgs += 1
#             if elapsed > 10 and elapsed < 11:
#                 #record number of images streamed in about 10secs
#                 avg_img_len = avg_img_len / cnt_streamed_imgs
#                 print("{} | Nbr_frames: {} - Elapased Time: {:.2f} | Average img length: {:.1f}]".format(n_recs, cnt_streamed_imgs, elapsed, avg_img_len) )
#                 summary.append( [cnt_streamed_imgs, elapsed, avg_img_len] )
#                 n_recs += 1
#                 #reset counters
#                 init = time.time()
#                 cnt_streamed_imgs = 0
#                 avg_img_len = 0
#             if n_recs == n_measurements:
#                 #Number of measurements
#                 test = False

#             #Write summary
#         with open("stream_perf_07.txt", "w") as file:
#             file.write("nbr_images, elapsed(sec), avg_img_size\n")
#             for record in summary:
#                 file.write("{}, {}, {}\n".format( record[0], record[1], record[2]))

#     finally:
#         connection.close()
#         server_socket.close()