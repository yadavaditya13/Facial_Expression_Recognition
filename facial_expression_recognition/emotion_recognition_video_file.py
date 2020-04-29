# This script will be used for emotion recognition in video file

# importing required packages

from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
import argparse
import imutils
import pickle
import random
import cv2
import os

# arguments parsing

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True, help="path to required input video file...")
ap.add_argument("-o", "--output", type=str, required=True, help="path to required output video file...")
ap.add_argument("-f", "--face", type=str, required=True, help="path to required face detector model...")
ap.add_argument("-m", "--model", type=str, required=True, help="path to ExpressNet model...")
ap.add_argument("-l", "--pickle", type=str, required=True, help="path to required label encoder file...")
ap.add_argument("-c", "--minconfidence", type=float, default=0.4, help="minimum confidence required for detection...")

args = vars(ap.parse_args())

# let's load our face detector pre-trained model

print("\n[INFO] Loading pre-trained Face-Detector Model from Disk...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
print("\n[INFO] Model loaded Successfully...")
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# let's load our ExpressionNet Trained model

print("\n[INFO] Loading ExpressionNet Trained Model and Label Encoder from Disk...")
model = load_model(args["model"])
le = pickle.loads(open(args["pickle"], "rb").read())
print("\n[INFO] Model and Label Encoder loaded Successfully...")

# let's define the emotions based on labels on which our model was trained
emotions = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# let's Load our video-file from disk

print("[INFO] Loading Video-File from Disk...")
vs = cv2.VideoCapture(args["input"])
writer = None

# calculating its length

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} :Total frames in Video".format(total))

except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# looping over the frames from Video
while vs.isOpened():

    # let's begin reading frames
    (grabbed, frame) = vs.read()

    # if no frame captured break the process
    if not grabbed:
        break

    frame = imutils.resize(frame, height=650, width=700)
    #frame_orig = frame.copy()

    # grabbing frame dimensions and converting it to a blob
    (h, w) = frame.shape[:2]

    # let's begin blobbing on our frames
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # passing blobs to our face-detector model

    #print("\n[INFO] Passing blobs to Face-Detector Model...\n")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    try:
        # looping over the detected faces in image
        for i in range(0, detections.shape[2]):

            # let's grab confidence of each detection
            confidence = detections[0, 0, i, 2]

            # filtering weak confidences
            if confidence > args["minconfidence"]:
                # let's grab detection box dimensions
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Now we need to get the ROIs from image and pass into our model for predictions
                # and pre-process it before passing through our model by first coverting it to grayscale

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # It's time for predictions

                preds = model.predict(face)[0]
                #print(preds)
                index = np.argmax(preds)
                #print(index)
                label = emotions[index]

                # Writing label and bounding box on the image
                label = "{}: {:.2f}%".format(label, preds[index] * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 0), 2)

        # Let's write the video frames to our disk...
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)

        writer.write(frame)

    except ValueError:
        print("[INFO] Whoops ... No Faces Detected...")

print("[INFO] Your video file is ready ... Jump to the desired directory of output!...")
# let's cleanup
vs.release()
writer.release()
cv2.destroyAllWindows()