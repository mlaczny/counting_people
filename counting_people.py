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

classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
writer = None
ct = CentroidTracker(maxDisappeared=60, maxDistance=50)
trackers = []
trackableObjects = {}
y_way = "Unidentified"
x_way = "Unidentified"
totalInVid = 0


def get_outputs_names(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs, class_number):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []
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
    global y_way
    global x_way
    global totalInVid
    inVid = 0

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            y_direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            inVid += 1
            if y_direction < 0:
                y_way = "UP"
            elif y_direction > 0:
                y_way = "DOWN"
            print(y_direction)

            x = [c[1] for c in to.centroids]
            x_direction = centroid[1] - np.mean(x)
            if x_direction > 0:
                x_way = "LEFT"
            elif x_direction < 0:
                x_way = "RIGHT"

        trackableObjects[objectID] = to
        idNumber = "ID {}".format(objectID)
        totalInVid = "{}".format(objectID)
        cv.putText(frame, y_way, (centroid[0] - 10, centroid[1] - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(frame, x_way, (centroid[0] - 10, centroid[1] - 35),
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
    for (i, (k, v)) in enumerate(info):


        text = "{}".format(v)
        if k == 'Total_in_video':

            cv.putText(frame, f'Total in video: {text}', (10, 55,),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if k == 'In_video':
            cv.putText(frame, f'In video now: {text}', (10, 95),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


winName = 'Counting objects'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

if not os.path.isfile(args.video):
    print("Input video file ", args.video, " doesn't exist")
    sys.exit(1)

cap = cv.VideoCapture(args.video)
outputFile = args.video[:-4] + '_output.mp4'
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                            (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    hasFrame, frame = cap.read()

    if not hasFrame:
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    outs = net.forward(get_outputs_names(net))

    postprocess(frame, outs, args.type_id)

    t, _ = net.getPerfProfile()
    time = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, time, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    vid_writer.write(frame.astype(np.uint8))
    cv.imshow(winName, frame)  # Wyświetlanie ramek video

