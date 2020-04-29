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

ap.add_argument("-i", "--input", type=str, required=True, help="path to required input image file...")
ap.add_argument("-f", "--face", type=str, required=True, help="path to required face detector model...")
ap.add_argument("-o", "--model", type=str, required=True, help="path to required ExpressionNet model file...")
ap.add_argument("-l", "--pickle", type=str, required=True, help="path to required label encoder file...")
ap.add_argument("-m", "--minconfidence", type=float, default=0.5, help="minimum confidence required for detection...")

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

# let's get our input image file

print("\n[INFO] Grabbing Input Image file...")
image = cv2.imread(args["input"])
orig = imutils.resize(image, height=650, width=700)
(h, w) = orig.shape[:2]
image = cv2.resize(orig, (300, 300))

# let's begin blobbing on our image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# let's try using various random colors for detections
np.random.seed(45)
COLORS = np.random.randint(0, 255, size=(25, 3), dtype="uint8")
#print(COLORS)

# passing blobs to our face-detector model

print("\n[INFO] Passing blobs to Face-Detector Model...\n")
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

            # Now we need to get the ROIs from image and pass into our model for predictions
            # and pre-process it before passing through our model

            face = orig[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # It's time for predictions

            preds = model.predict(face)[0]
            print(preds)
            index = np.argmax(preds)
            print(index)
            label = emotions[index]

            # Writing label and bounding box on the image
            label = "{}: {:.2f}%".format(label, preds[index] * 100)
            print(label)
            color = random.choice(COLORS)
            print(color)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (int(color[0]), int(color[1]), int(color[2])), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (int(color[0]), int(color[1]), int(color[2])), 2)

    cv2.imshow("Face Expression: ", orig)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        # let's cleanup
        cv2.destroyAllWindows()

except ValueError:
    print("[INFO] Whoops ... No Faces Detected...")