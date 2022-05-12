import cv2
import time
import random
import numpy as np
from threading import Thread
import math
import serial

def cal_dist_pythagoras(bbox_center, camera_center):
	camera_x_center, camera_y_center = camera_center
	bbox_x_center, bbox_y_center = bbox_center

	dist = math.sqrt((camera_x_center - bbox_x_center) ** 2 + (camera_y_center - bbox_y_center) ** 2)

	return dist


def robot_move(dist_from_center_list, bbox_center_list, camera_center, frame):
	min_value = min(dist_from_center_list)
	min_index = dist_from_center_list.index(min_value)

	min_dist_bbox = bbox_center_list[min_index]
	obj_x, obj_y = min_dist_bbox
	camera_x, camera_y = camera_center

	x_dist = camera_x - obj_x
	y_dist = camera_y - obj_y

	cv2.arrowedLine(frame, (camera_x, camera_y), (obj_x, obj_y), (0, 0, 255), 2, 8, 0, 0.1)

	if (x_dist < -15) and (y_dist > 15): # Move right + up
		# print('Object in Q1')
		text = f'X1{abs(x_dist)},Y1{abs(y_dist)}*'
		print('Object in Q1 : ' + text)
		arduino.write(text.encode())
	elif (x_dist < -15) and (y_dist < -15):  # Move right + down
		# print('Object in Q2')
		text = f'X1{abs(x_dist)},Y0{abs(y_dist)}*'
		print('Object in Q2 : ' + text)
		arduino.write(text.encode())
	elif (x_dist > 15) and (y_dist < -15):  # Move left + down
		# print('Object in Q3')
		text = f'X0{abs(x_dist)},Y0{abs(y_dist)}*'
		print('Object in Q3 : ' + text)
		arduino.write(text.encode())
	elif (x_dist > 15) and (y_dist > 15):  # Move left + up
		# print('Object in Q4')
		text = f'X0{abs(x_dist)},Y1{abs(y_dist)}*'
		print('Object in Q4 : ' + text)
		arduino.write(text.encode())
	elif (x_dist < -15) and abs(y_dist < 15):  # Move right
		# print('Object in Q4')
		text = f'X1{abs(x_dist)},Y1{0}*'
		print('Object in X+ : ' + text)
		arduino.write(text.encode())
	elif (x_dist > 15) and abs(y_dist < 15):  # Move left
		# print('Object in Q4')
		text = f'X0{abs(x_dist)},Y1{0}*'
		print('Object in X- : ' + text)
		arduino.write(text.encode())
	elif abs(x_dist < 15) and (y_dist > 15):  # Move up
		# print('Object in Q4')
		text = f'X1{0},Y1{abs(x_dist)}*'
		print('Object in Y+ : ' + text)
		arduino.write(text.encode())
	elif abs(x_dist < 15) and (y_dist < -15):  # Move down
		# print('Object in Q4')
		text = f'X1{0},Y0{abs(x_dist)}*'
		print('Object in Y- : ' + text)
		arduino.write(text.encode())	
	else:
		print("center OK")

	# if abs( camera_x - obj_x) > 15:
	# 	if camera_x - obj_x >= 0:
	# 		print("move >> left")
	# 		arduino.write('X1*'.encode())

	# 	else:
	# 		print("move >> right")
	# 		arduino.write('X0*'.encode())

	# else:
	# 	if abs( camera_y - obj_y) > 15:
	# 		if camera_y - obj_y >= 0:
	# 			arduino.write('Y0*'.encode())
	# 			print("move >> up")
	# 		else:
	# 			arduino.write('Y1*'.encode())
	# 			print("move >> down")
	# 	else:
	# 		print("center OK")


def pluse():
	while cap.isOpened():
		if main_fn:
			GPIO.output(pluse_pin, GPIO.HIGH)
			time.sleep(0.5)
			GPIO.output(pluse_pin, GPIO.LOW)
			time.sleep(0.5)
		else:
			pass
	cap.release()


def object_detection():
	while cap.isOpened():
		prev_time = time.time()

		ret, frame = cap.read()

		classname = []
		dist_from_center_list = []
		bbox_center_list = []

		if not ret:
			print('frame read error!!!!')
			break

		if camera_calibrate:
			undis = cv2.undistort(frame, oldMtx, coef, None, newMtx)
			x_roi, y_roi, w_roi, h_roi = roi
			frame = undis[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

		dimensions = frame.shape
		# print(dimensions)
		camera_center = (int(dimensions[1]/2), int(dimensions[0]/2))

		classIds, scores, boxes = model.detect(frame, confThreshold=thres, nmsThreshold=0.25)

		for (classId, score, box) in zip(classIds, scores, boxes):
			#if classes[classId[0]] == 'person':
				cv2.rectangle(frame, (box[0], box[1]),
							  (box[0] + box[2], box[1] + box[3]),
							  color[classId], thickness = bbox_thickness)

				cv2.circle(frame,(int(box[0] + (box[2] / 2)) ,int(box[1] + (box[3] / 2))), 5, (0, 0, 0), 2)

				text = f'{classes[classId]}: {score*100:.2f}'
				(label_width, label_height), baseline = cv2.getTextSize(
													  text,
													  cv2.FONT_HERSHEY_SIMPLEX,
													  label_size,
													  thickness = label_thickness)
				cv2.putText(frame, text,
							(box[0] + 3, box[1] + label_height + baseline ),
							cv2.FONT_HERSHEY_SIMPLEX, label_size,
							color[classId], thickness = label_thickness)

				bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
				bbox_center = [int(box[0] + (box[2] / 2)) ,int(box[1] + (box[3] / 2))]

				bbox_center_list.append(bbox_center)

				dist_from_center = cal_dist_pythagoras(bbox_center, camera_center)

				dist_from_center_list.append(dist_from_center)

		if not len(dist_from_center_list) == 0:
			robot_move(dist_from_center_list, bbox_center_list, camera_center, frame)

		cv2.circle(frame, (camera_center[0], camera_center[1]), 20, (0, 255, 0), 2)

		fps = round(1/(time.time() - prev_time), 2)

		(label_width, label_height), baseline = cv2.getTextSize('K - VISION',cv2.FONT_HERSHEY_SIMPLEX,
									1, thickness = 2)

		image_bordered = cv2.copyMakeBorder(src=frame, top=(4 + label_height), bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT)

		cv2.putText(image_bordered, 'K - VISION', (2, 2 + label_height),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 213, 48), thickness = 2)

		(label_width, label_height), baseline = cv2.getTextSize("FPS :  {}".format(fps),cv2.FONT_HERSHEY_SIMPLEX,
									1, thickness = 2)

		cv2.putText(image_bordered, f"FPS :  {fps}", (frame_width - label_width - 2, 2 + label_height),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness = 2)

		cv2.imshow('Image', image_bordered)

		if cv2.waitKey(20) == 27:
			break

		# time.sleep(0.5)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	arduino = serial.Serial(port = 'COM4', baudrate = 115200, timeout=0.01)

	camera_calibrate = True

	frame_width = 1280
	frame_height = 960

	#Cam param for undistort pic
	if camera_calibrate:

		oldMtx = np.load('cameraMatrix.npy')	#change when use new camera
		coef = np.load('distortionCoef.npy')	#change when use new camera

		newMtx, roi = cv2.getOptimalNewCameraMatrix(oldMtx, coef, (frame_width,frame_height), 1, (frame_width,frame_height))

	thres = 0.8

	#Drawing property
	bbox_thickness = 2
	label_size = 0.5
	label_thickness = 2
	fps_size = 0.5
	fps_thickness = 1

	color = []

	#Video property
	cap = cv2.VideoCapture(1)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

	with open('obj.names', 'r') as f:
		classes = f.read().splitlines()

	random.seed(120)

	for i in range(len(classes)):
		color.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))

	net = cv2.dnn.readNetFromDarknet('yolov4-tiny-custom.cfg','yolov4-tiny-custom_last.weights')

	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

	Thread(target=object_detection).start()
	#Thread(target=pluse).start()
