from picamera2 import Picamera2, Preview
import time
import cv2
import matplotlib.pyplot as plt
 
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(buffer_count=2)
camera_config2 = picam2.create_preview_configuration()
picam2.configure(camera_config2)
picam2.start_preview(Preview.QT)
picam2.start()
time.sleep(1)
picam2.capture_file("test.jpg")
array = picam2.capture_array("main")
plt.ion()
# cv2.imshow("a",array)
# cv2.waitKey(0)
while True:
    time1 = time.time()
    array = picam2.capture_array("main")
    print(time.time()-time1)

    plt.subplot(1,2,1), plt.imshow(array, interpolation='nearest')  
    plt.pause(0.001)
    plt.show()