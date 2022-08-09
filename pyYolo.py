import numpy as np
import time
import cv2

# https://cloudxlab.com/blog/object-detection-yolo-and-python-pydarknet/


def main(image, FileTime):
	# INPUT_FILE='/home/sunny/wb_data/{}_0_Color.jpg'.format(FileTime)
	OUTPUT_FILE='/home/sunny/wb_data/{}_Pre.jpg'.format(FileTime)
	LABELS_FILE='/home/sunny/darknet/data/coco.names'
	# CONFIG_FILE='/home/sunny/darknet/wb/yolov3_wb_2k10.cfg'
	# WEIGHTS_FILE='/home/sunny/darknet/wb/yolov3_wb_2k10_final.weights'
	CONFIG_FILE = '/home/sunny/darknet/wb/yolov3_wb.cfg'
	WEIGHTS_FILE = '/home/sunny/darknet/wb/yolov3_wb_kinect_final.weights'

	CONFIDENCE_THRESHOLD=0.85

	LABELS = open(LABELS_FILE).read().strip().split("\n")

	# np.random.seed(2)
	# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	# 	dtype="uint8")
	COLORS = [[10, 10, 220], [220, 10, 10]]
	# print(COLORS)

	net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

	# image = cv2.imread(INPUT_FILE)
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	print("[INFO] YOLO took {:.3f} seconds".format(end - start))


	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
		CONFIDENCE_THRESHOLD)

	wbmid = []
	water = []
	yenhi = []
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
			if LABELS[classIDs[i]] == "wbmid":
				wbmid.append([x + w // 2, y + h // 2])
			elif LABELS[classIDs[i]] == "water":
				water.append([x + w // 2, y + h // 2])
			# elif LABELS[classIDs[i]] == "yenhi":
			# 	if w > h:
			# 		yenhi.append([x + w // 2, y + h // 2, 0])
			# 	else:
			# 		yenhi.append([x + w // 2, y + h // 2, 1])
			# cut_image = image[int(y * 1):int((y + h) * 1), int(x * 1):int((x + w) * 1)]
			# cut_image = cv2.cvtColor(cut_image, cv2.COLOR_RGB2GRAY)
			# blur = cv2.GaussianBlur(cut_image, (5, 5), 0)
			# ret3, OTSU_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			# cv2.imwrite('/home/sunny/wb_data/cutimg/{}1_Cut_{}_{}.jpg'.format(FileTime, x, y), OTSU_image)



	# show the output image
	cv2.imwrite(OUTPUT_FILE, image)
	print("yolo wbmid:", wbmid)
	print("yolo water:", water)
	print("yolo yenhi:", yenhi)
	return wbmid, water
	# return yenhi



if __name__ == "__main__":
	start = time.time()
	image = cv2.imread("/home/sunny/wb_data/1314.jpg")
	# print(image[0])
	main(image, 123)
	end = time.time()
	print("Time:", round(end - start, 2))