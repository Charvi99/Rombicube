import cv2

capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 4032
height = 3040
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capture.set(cv2.CAP_PROP_EXPOSURE, -13) 

while True:
    capture.grab()
    ret, frame = capture.retrieve(0)
    if not ret:
        # if not True:
        break
    # rombiCube.estimatePose(frame=frame)
    cv2.imshow('Estimated Pose', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()