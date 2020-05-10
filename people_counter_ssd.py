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
# initalisation
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
model = cv2.dnn.readNetFromCaffe("mobilenet_ssd\MobileNetSSD_deploy.prototxt", "mobilenet_ssd\MobileNetSSD_deploy.caffemodel")
vs = cv2.VideoCapture(0)
TIME = 5
CONFIDENCE = 0.8
counter = 50
timer = 0
currentID = 0
redetected = 0
width = []
height = []
av_hght = 50
av_wdth = 50
# portal (x, x, y, y)
portal = (220, 290, 0, 80)
trackable_objects = {}
class trackableobject:
    def get_bbox(self):
        return self.bbox
    def __init__ (self, objectID, bbox, tracker):
        self.objectID = objectID
        self.bbox = bbox
        self.counted = False
        self.detected = True
        self.tracker = tracker
        self.start = 'none'
while True:
	# get next frame from video source
    (grabbed, frame) = vs.read()
    (h, w) = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # DETECTION (every x frames new detection phase)
    if timer == TIME:
        for objectID in trackable_objects:
            trackable_objects[objectID].detected = False
        # reset timer
        timer = 0
		# convert the frame to a blob
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
		# pass the blob through the network and obtain the detections
        model.setInput(blob)
        detections = model.forward()
		# loop  detections
        for i in np.arange(0, detections.shape[2]):
			# find confidence
            confidence = detections[0, 0, i, 2]
			# filter out bad confidence
            if confidence > CONFIDENCE:
				# only accept people class
                idx = int(detections[0, 0, i, 1])
                if idx != 15:
                    continue
                redetected = False
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
                    if (cent_x_ - int(av_wdth)) < cent_x < (cent_x_ + int(av_wdth)) and \
                        (cent_y_ - int(av_hght)) < cent_y < (cent_y_ + int(av_hght)):
                        currentID = objectID
                        t = trackable_objects[objectID].tracker
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        t.start_track(rgb, rect)
                        trackable_objects[objectID].detected = True
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
                            trackable_objects[currentID].detected = True
                            break
        # delete undetected objects
        del_track_obj = {**trackable_objects}
        for objectID in del_track_obj:
            if del_track_obj[objectID].detected == False:
                del trackable_objects[objectID]
        # finding average width and height
        if len(trackable_objects) > 0:
            width = [0]*len(trackable_objects)
            height = [0]*len(trackable_objects)
            for i, objectID in enumerate(trackable_objects):
                boundingbox = trackable_objects[objectID].get_bbox()
                width[i] = int(boundingbox[2])
                height[i] = int(boundingbox[3])
            sum_ = 0
            for x in width:
                sum_ += x
            av_wdth = (sum_ / (len(width)*3))
            sum_ = 0
            for x in height:
                sum_ += x
            av_hght = (sum_ / (len(height)*3))
    # TRACKING
    else:
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
            # update dictionary
            trackable_objects[objectID].bbox = (startX, startY, abs(startX-endX), abs(startY-endY))   
    # COUNTING
    for objectID in trackable_objects:
        # find centroid
        bbox = trackable_objects[objectID].get_bbox()
        cent_x = int(bbox[0]) + int(bbox[2]/2)
        cent_y = int(bbox[1]) + int(bbox[3]/2)
        # assign starting location
        if trackable_objects[objectID].start == 'none':
            if portal[0] < cent_x < portal[1] and portal[2] < cent_y < portal[3]:
                trackable_objects[objectID].start = 'out'
            else:
                trackable_objects[objectID].start = 'in'
        # if not counted check location
        elif trackable_objects[objectID].counted == False:
            if portal[0] < cent_x < portal[1] and portal[2] < cent_y < portal[3]:
                if trackable_objects[objectID].start == 'in':
                    counter -= 1
                    print('out', objectID)
                    counter = max(0, counter)
                    trackable_objects[objectID].counted = True
            else:
                if trackable_objects[objectID].start == 'out':
                    counter += 1
                    print('in', objectID)
                    trackable_objects[objectID].counted = True
    # display                    
    for objectID in trackable_objects:
        bbox = trackable_objects[objectID].get_bbox()
        startX = int(bbox[0])
        startY = int(bbox[1])
        endX = int(bbox[0]+bbox[2])
        endY = int(bbox[1]+bbox[3])                
        # make coords positive
        startX = max(0, startX)
        startY = max(0, startY)
        # display everything
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
        cv2.putText(frame, ('ID: {}'.format(str(objectID))), (startX, startY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
        cv2.putText(frame, ('Start: {}'.format(str(trackable_objects[objectID].start))), (startX, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
        cv2.putText(frame, ('Counted: {}'.format(str(trackable_objects[objectID].counted))), (startX, startY + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
        blur_region = frame[startY:endY, startX:endX]
        blur = cv2.GaussianBlur(blur_region, (51,51), 0)       
        frame[startY:endY, startX:endX] = blur   
    # portal box
    cv2.rectangle(frame, (portal[0], portal[2]), (portal[1], portal[3]), (0, 0, 0), 4)
    # show count
    cv2.rectangle(frame, (0, h), (640, int(h-20)), (0, 0, 0), -1)
    cv2.putText(frame, 'COUNT = {}'.format(counter), (int((w/2)-100), int(h-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # show the output frame
    cv2.imshow("Frame", frame)
	# if the esc key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == 'q':
        start = True
    #increment timer
    timer = timer + 1
# release
cv2.destroyAllWindows()
vs.release()