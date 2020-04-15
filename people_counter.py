"""
ENGR 300 - People detection and tracking

Created December 6 2019
Last updated March 30 2020

Author Thomas Lumborg
"""

# module import
import numpy as np
import dlib
import cv2

# class label initalisation
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# load model
model = cv2.dnn.readNetFromCaffe("mobilenet_ssd\MobileNetSSD_deploy.prototxt", "mobilenet_ssd\MobileNetSSD_deploy.caffemodel")
# initialise video stream
vs = cv2.VideoCapture(0)

# initialise lists and ints
frames_between_detection = 30
counter = 0
timer = 0
currentID = 0
redetected = 0

class trackableobject:
    def get_bbox(self):
        return self.bbox
    def __init__ (self, objectID, bbox, tracker):
        self.objectID = objectID
        self.bbox = bbox
        self.counted = False    
        self.detected = True
        self.tracker = tracker
trackable_objects = {}

# loop over frames from the video file stream
while True:
	# grab the next frame from the video file
    (grabbed, frame) = vs.read()
    #convert to RGB from BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # object detection (every x frames new detection)
    if timer == frames_between_detection:
        for objectID in trackable_objects:
            trackable_objects[objectID].detected = False
        # print to console
        print('count = ', counter)
        # reset timer
        timer = 0
		# grab the frame dimensions and convert the frame to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
		# pass the blob through the network and obtain the detections
        model.setInput(blob)
        detections = model.forward()
		# loop over the detections
        for i in np.arange(0, detections.shape[2]):
			# find confidence
            confidence = detections[0, 0, i, 2]
			# filter out bad confidence
            if confidence > 0.6:
                redetected = False
				# only accept people class
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
				# compute the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bbox = (startX, startY, abs(startX-endX), abs(startY-endY))
                cent_x = startX + (endX-startX)/2
                cent_y = startY + (endY-startY)/2
                # CHECK EXISTING OBJECTS
                for objectID in trackable_objects:
                    bbox_ = trackable_objects[objectID].get_bbox()
                    cent_x_ = int(bbox_[0]) + (int(bbox_[2]))/2
                    cent_y_ = int(bbox_[1]) + (int(bbox_[3]))/2
                    if (cent_x_ - 50) < cent_x < (cent_x_ + 50) and (cent_y_ - 50) < cent_y < (cent_y_ + 50):
                        currentID = objectID
                        redetected = True
                        break
                # ADD NEW OBJECTS
                if redetected == False:
                    to_dict_length = len(trackable_objects) + 1
                    for x in range(0, to_dict_length):
                        if x in trackable_objects:
                            continue
                        else:
                            currentID = x
                            t = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            t.start_track(rgb, rect)
                            trackable_objects[currentID] = trackableobject(currentID, bbox, t)
                            break
                trackable_objects[currentID].detected = True
                # draw the bounding box and ID
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, str(currentID), (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	# tracking and counting system
    else:
        # delete objects
        for objectID in trackable_objects:
            if trackable_objects[objectID].detected == False:
                del trackable_objects[objectID]
                break
		# loop over each of the objects
        for objectID in trackable_objects:
			# update the tracker and grab the position of the object
            t = trackable_objects[objectID].tracker
            t.update(rgb)
            pos = t.get_position()
			# unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # find new centroid
            cent_x = startX + (endX-startX)/2
            cent_y = startY + (endY-startY)/2
            # find old centroid
            bbox_ = trackable_objects[objectID].get_bbox()
            cent_x_ = int(bbox_[0]) + (int(bbox_[2]))/2
            cent_y_ = int(bbox_[1]) + (int(bbox_[3]))/2
            # update dictionary
            trackable_objects[objectID].bbox = (startX, startY, abs(startX-endX), abs(startY-endY))
            # COUNTING
            # check if already counted and time after last detection phase
            if trackable_objects[objectID].counted == False:
                # check if moving right
                if cent_x_ < cent_x:
                    trackable_objects[objectID].counted = True
                    counter = counter + 1
                # check if moving left
                if cent_x_ > cent_x:
                    trackable_objects[objectID].counted = True
                    counter = counter - 1
                    if counter < 0:
                        counter = 0
			# draw the bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, str(objectID), (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	# show dividing line in middle
    cv2.line(frame, (320, 0), (320, 480), (0, 255, 0), 4)
    # show count in corner
    cv2.putText(frame, 'Count = {}'.format(counter), (5, 475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # show the output frame
    cv2.imshow("Frame", frame)
	# if the esc key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    #increment timer
    timer = timer + 1
# release
cv2.destroyAllWindows()
vs.release()