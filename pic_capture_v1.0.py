import cv2
import numpy as np
import time
import os

def nothing(x):
    pass

##### camera properties #####
cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

time_last_capture = 0

track_bar = 'Track Bar'
cv2.namedWindow(track_bar)
cv2.createTrackbar('time cap', track_bar, 60, 120, nothing)
cv2.createTrackbar('start cap', track_bar, 0, 1, nothing)

##### capture save directory #####
directory = r'C:\Users\Lenovo\Documents\python program\robot_pic_only'

os.chdir(directory)

while cap1.isOpened :
    ret1, frame1 = cap1.read()

    cap_interval = cv2.getTrackbarPos('time cap', track_bar)
    start_cap = cv2.getTrackbarPos('start cap', track_bar)

    cv2.imshow('frame1', frame1)

    if start_cap == 1:
        if time.time() - time_last_capture >= cap_interval:
            time_last_capture = time.time()

            cv2.imwrite(filename1, frame1)

    if cv2.waitKey(1) == 27:
        break

cap2.release()
cv2.destroyAllWindows()
