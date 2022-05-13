import cv2
import time
import random
import numpy as np
import threading
import serial
import sys
from PIL import Image, ImageTk
import tkinter as tk

def bbox_drawing(frame, classId, score, bbox):
    cv2.rectangle(frame, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), bbox_color[classId],
                  thickness = bbox_thickness)
    label = f'{classes[classId]}: {score*100:.2f}'
    (label_width, label_height), baseline = cv2.getTextSize(
                                          label,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          label_size,
                                          thickness = label_thickness)
    cv2.putText(frame, label,
                (bbox[0] + 3, bbox[1] + label_height + baseline ),
                cv2.FONT_HERSHEY_SIMPLEX, label_size,
                bbox_color[classId], thickness = label_thickness)


def object_detected_drawing(frame, dict, height, width):
    text = 'Object Detected'
    (label_w, label_h), baseline = cv2.getTextSize(text, 
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.75, thickness = 2)

    cv2.putText(frame, text, (width - label_w - 10, 2 + label_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 0), thickness = 2)
    cv2.putText(frame, text, (width - label_w - 10, 2 + label_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 255), thickness = 1)

    x = width - label_w - 10
    y = 2 + label_h
    for key in dict:
        text = f'{key} : {dict[key]}'
        (label_w, label_h), baseline = cv2.getTextSize(
                                          text,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.75,
                                          thickness = 1)
        cv2.putText(frame, text, (x, y + label_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 0), thickness = 2)
        cv2.putText(frame, text, (x, y + label_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 255), thickness = 1)

        y = y + label_h + 5

    cv2.putText(frame, 'Status : ', (2, height - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), thickness = 2)
    cv2.putText(frame, 'Status : ', (2, height - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), thickness = 1)


def ui_drawing(frame):
    (label_w, label_h), baseline = cv2.getTextSize('K - VISION', 
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1, thickness = 2)

    frame = cv2.copyMakeBorder(src=frame, top=(10 + label_h), 
            bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT)
    
    cv2.putText(frame, 'K - VISION', (2, 4 + label_h),
               cv2.FONT_HERSHEY_SIMPLEX, 1,
               (200, 213, 48), thickness = 2)

    return frame


def object_detection():
    cv2.namedWindow('Video Capture', cv2.WINDOW_NORMAL)

    while camera.isOpened():

        start = time.time()

        class_name_list = []
        det_object_list = []

        ret, frame = camera.read()

        if not ret:
            print('frame read error!!')
            break

        if camera_calibrate:
            undis = cv2.undistort(frame, oldMtx, coef, None, newMtx)
            x_roi, y_roi, w_roi, h_roi = roi
            frame = undis[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

        frame_h, frame_w, _ = frame.shape

        classIds, scores, boxes = model.detect(frame,
                                  confThreshold = conf_thres,
                                  nmsThreshold = nms_thres)

        for classId in classIds:
            class_name_list.append(classes[classId])

        class_name_list_rm_dup = set(class_name_list)

        class_dict = { name : class_name_list.count(name) 
                     for name in (class_name_list_rm_dup) }

        # print(class_dict)

        for (classId, score, box) in zip(classIds, scores, boxes):
            bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            bbox_drawing(frame, classId, score, bbox)

        object_detected_drawing(frame, class_dict, frame_h, frame_w)

        frame = ui_drawing(frame)

        cv2.imshow('Video Capture', frame)

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

def set_up_properties():
    global camera
    global classes
    global model
    global oldMtx
    global coef
    global newMtx
    global roi

    print('waiting for camera')

    camera = cv2.VideoCapture(cam_index)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    print('camera connected!!')

    with open(names_file, 'r') as f:
        classes = f.read().splitlines()

    random.seed(120)

    for i in range(len(classes)):
        bbox_color.append((random.randint(0,255), random.randint(0,255),
                    random.randint(0,255)))

    net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    if camera_calibrate:
        oldMtx = np.load(camera_matrix_file)
        coef = np.load(distortion_coef_file)

        newMtx, roi = cv2.getOptimalNewCameraMatrix(oldMtx, coef,
                      (cam_width,cam_height), 1,
                      (cam_width,cam_height))

    object_detection()

if __name__ == '__main__':
    # arduino = serial.Serial(port = 'COM4', baudrate = 115200, timeout=0.01)

    bbox_color = []

    bbox_thickness = 2
    label_size = 0.5
    label_thickness = 2

    cam_index = 1
    cam_width = 1280
    cam_height = 720

    names_file = 'coco.names' # .names
    cfg_file = 'yolov4-tiny.cfg' # .cfg
    weights_file = 'yolov4-tiny.weights' # .weights

    camera_calibrate = False
    camera_matrix_file = '' # .npy
    distortion_coef_file = '' # .npy

    use_cuda = True

    conf_thres = 0.5
    nms_thres = 0.3

    threading.Thread(target = set_up_properties).start()