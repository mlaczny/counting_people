import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject

# Initialize the parameters
confThreshold = 0.6  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection')

parser.add_argument('--video', default='vid/test_3.mp4', help='Path to video file.')
parser.add_argument('--type_id', default=0, help='Object type', type=int)
args = parser.parse_args()

# Load names of classes
classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
writer = None
ct = CentroidTracker(maxDisappeared=60, maxDistance=50)
trackers = []
trackableObjects = {}
way = "Unidentified"
totalInVid = 0


# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, class_number):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class 0 = "person", 1 = "bike"
        if classIds[i] == class_number:  # ludzie
            rects.append((left, top, left + width, top + height))
            objects = ct.update(rects)
            counting(objects)


def counting(objects):
    global way
    global totalInVid
    inVid = 0

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID

        to = trackableObjects.get(objectID, None)
        # print(totalInVid)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            #print(direction)
            to.centroids.append(centroid)
            inVid += 1
            if direction < 0:
                way = "UP"
            elif direction > 0:
                way = "DOWN"

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        idNumber = "ID {}".format(objectID)
        totalInVid = "{}".format(objectID)
        cv.putText(frame, way, (centroid[0] - 10, centroid[1] - 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(frame, idNumber, (centroid[0] - 10, centroid[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Total_in_video", int(totalInVid) + 1),
        ("In_video", inVid),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):

        text = "{}".format(v)
        if k == 'Total_in_video':
            cv.putText(frame, f'Total in video: {text}', (10, 55),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if k == 'In_video':
            cv.putText(frame, f'In video now: {text}', (10, 95),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


# Process inputs
winName = 'Counting objects'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# Open the video file
if not os.path.isfile(args.video):
    print("Input video file ", args.video, " doesn't exist")
    sys.exit(1)

cap = cv.VideoCapture(args.video)
outputFile = args.video[:-4] + '_output.mp4'
# Get the video writer initialized to save the output video
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                            (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    '''
    image – This is the image that we want to preprocess (for our model).
    scalefactor – scale factor basically multiplies(scales) our image channels. 
    size – this is the target size that we want our image to become. 
    mea n– this is the mean subtracting values. You can input a single value or a 3-item tuple for each channel RGB, 
    it will basically subtract the mean value from all the channels accordingly, this is done to normalize our pixel 
    values.
    '''
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, args.type_id)

    t, _ = net.getPerfProfile()
    time = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, time, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Write the frame with the detection boxes

    vid_writer.write(frame.astype(np.uint8))
    cv.imshow(winName, frame)  # Wyświetlanie ramek video

# ludzie
# Usage:  python counting_people.py --video /Users/m.laczny/PycharmProjects/People_counting_basic/vid/test.mp4 --type_id 1
# auta
# Usage:  python counting_people.py --video /Users/m.laczny/PycharmProjects/People_counting_basic/vid/test_2.mp4 --type_id 2
# uczelnia
# Usage:  python counting_people.py --video /Users/m.laczny/PycharmProjects/People_counting_basic/vid/test_2.mp4
