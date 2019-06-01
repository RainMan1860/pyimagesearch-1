from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np 
import argparse
import imutils
import time
import dlib
import cv2

# define the eye aspect ratio according to formula
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2 * C)

# Create the argument parse. Input arugments are path to facial landmark and
# path to input video file 
ap = argparse.ArgumentParser()

# This is the path to dlib's pre-trained facial landmark detector
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help='path to input video file')
args = vars(ap.parse_args())

# Set the eye aspect ratio threshold and for how many frames
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and total number of blinks
COUNTER = 0
TOTAL = 0

# initialize face detector
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# dlib uses a pre-trained face detector which is based on a modification to 
# the Histogram of Oriented Gradients + Linear SVM method for object detection

# grab the indexes of the starting and ending array slice values for the
# facial landmarks of the eyes 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start webcam video stream 
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(2)

while True:
	# grab the frame, resize, and convert to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces 
	rects = detector(gray, 0)

	# apply facial landmark detection for each face in the frame
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract left and right eye coordinates
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		# get eye aspect ratio-
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2

		# Draw a convex hull around the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# Check if EAR is below blink threshold
		if ear < EYE_AR_THRESH:
			COUNTER += 1
		# check if eyes have been open or eyes have just opened after blink
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			COUNTER = 0

		# draw the total number of blinks. Don't put EAR because it looks weird
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# Show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# exit if q key is pressed
	if key == ord("q"):
		break

# clean up the windows and stop the webcam
cv2.destroyAllWindows()
vs.stop()