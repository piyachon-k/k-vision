import cv2
import time
import random
import numpy as np
import threading
import serial
import sys

def two_layer_text(frame, text, b_color, 
                   f_color, b_thick, f_thick, 
                   text_size, pos_x, pos_y):

    cv2.putText(frame, text, (pos_x, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX, text_size,
                b_color, thickness = b_thick)
    cv2.putText(frame, text, (pos_x, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX, text_size,
                f_color, thickness = f_thick)


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


def object_detected_drawing(frame, dict, height, width, checking_status):
    text = 'OBJECT DETECTED'
    (label_w, label_h), baseline = cv2.getTextSize(text, 
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.75, thickness = 3)
    two_layer_text(frame, text, color['black'], color['yellow'], 
                   3, 1, 0.75, (width - label_w - 10), (2 + label_h))

    y = 2 + label_h + baseline
    for key in dict:
        text = f'{key} : {dict[key]}'
        (label_w, label_h), baseline = cv2.getTextSize(
                                          text,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.75,
                                          thickness = 3)
        two_layer_text(frame, text, color['black'], color['yellow'],
                       3, 1, 0.75, (width - label_w - 5), (y + label_h + 2))
        y = y + label_h + 2 + baseline
    
    # text = f'Status : {checking_status}'
    # (label_w, label_h), baseline = cv2.getTextSize(
    #                                             text,
    #                                             cv2.FONT_HERSHEY_SIMPLEX,
    #                                             1.5,
    #                                             thickness = 6)
    
    # if checking_status == 'NG':
    #     two_layer_text(frame, text, color['black'], color['yellow'],
    #                     6, 3, 1.5, 2, (2 + label_h))
    # elif checking_status == 'OK':
    #     two_layer_text(frame, text, color['black'], color['green'],
    #                     6, 3, 1.5, 2, (2 + label_h))


def ui_drawing(frame, fps, width):
    (label_w, label_h), baseline = cv2.getTextSize('K - VISION', 
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1, thickness = 2)

    frame = cv2.copyMakeBorder(src = frame, top = (10 + label_h), 
            bottom = 0, left = 0, right = 0, borderType = cv2.BORDER_CONSTANT)
    
    cv2.putText(frame, 'K - VISION', (2, 4 + label_h),
               cv2.FONT_HERSHEY_SIMPLEX, 1,
               (200, 213, 48), thickness = 2)

    (label_w, label_h), baseline = cv2.getTextSize(f'FPS : {fps:.2f}', 
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1, thickness = 2)
    cv2.putText(frame, f'FPS : {fps:.2f}', (width - 2 - label_w, 4 + label_h),
               cv2.FONT_HERSHEY_SIMPLEX, 1,
               (200, 213, 48), thickness = 2)

    return frame


def serial_print(class_name_list):
    for class_name in class_name_list:
        if class_name == "CH36":
            text = '33'
            arduino.write(text.encode())
        elif class_name == "TMC60D":
            text = '34'
            arduino.write(text.encode())
        # elif class_name == "":
        #     text = '35'
        #     arduino.write(text.encode())


def object_detection():

    while camera.isOpened():
    # while True:

        start = time.time()

        cv2.namedWindow('Video Capture', cv2.WINDOW_NORMAL)

        class_name_list = []
        det_object_list = []
        detect_status = []

        ret, frame = camera.read()
        # ret = True
        # frame = np.zeros((cam_height, cam_width, 3), np.uint8)
        # frame[:,0:cam_width] = (255,255,255)

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

        detected_dict = { name : class_name_list.count(name) 
                     for name in (class_name_list_rm_dup) }

        # for key in things_to_det:
        #     if key in detected_dict:
        #         if things_to_det[key] == detected_dict[key]:
        #             detect_status.append(True)
        #         else:
        #             detect_status.append(False)
        #     else:
        #         detect_status.append(False)

        # if False in detect_status:
        #     checking_status = 'NG'
        # else:
        #     checking_status = 'OK'

        # print(detected_dict)

        for (classId, score, box) in zip(classIds, scores, boxes):
            bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            bbox_drawing(frame, classId, score, bbox)

        serial_print(class_name_list_rm_dup)
    
        object_detected_drawing(frame, detected_dict, frame_h, frame_w, checking_status)

        fps = 1/(time.time() - start)

        frame = ui_drawing(frame, fps, frame_w)

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
    ##### set up i/o #####
    # arduino = serial.Serial(port = 'COM4', baudrate = 115200, timeout=0.01)

    bbox_color = []
  
    color = {
        'blue': (255, 0, 0), 'green': (0, 255, 0),  'red': (0, 0, 255),
        'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0),
        'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (125, 125, 125),
        'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

    bbox_thickness = 2
    label_size = 0.5
    label_thickness = 2

    ##### camera properties #####
    cam_index = 1
    cam_width = 1280
    cam_height = 720

    ##### weight file #####
    names_file = 'coco.names' # .names
    cfg_file = 'yolov4-tiny.cfg' # .cfg
    weights_file = 'yolov4-tiny.weights' # .weights

    ##### camera calibrate #####
    camera_calibrate = False
    camera_matrix_file = '' # .npy
    distortion_coef_file = '' # .npy

    ##### back end [True : cuda / False : cpu] #####
    use_cuda = True

    ##### detection threshold #####
    conf_thres = 0.5
    nms_thres = 0.3

    ##### list of object you want to detect #####
    # things_to_det = {'person': 1, 'surfboard': 1}

    threading.Thread(target = set_up_properties).start()