# deque is a double ended queue. Hybrid linear structure provides all the 
# capabilities of a stack and queue in the same data structure. This structure
# will allow us to odraw where the ball has been in the past
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# Create argument parse
ap = argparse.ArgumentParser()
# use webcam if optional video argument is not entered
ap.add_argument("-v", "--video", help="path to the optional video file")

# maximum size of the deque. This will lead to a shorter tail
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green" balll in the HSV color
# space, then initialize the list of tracked points
colorLower = (29, 86, 6)
colorUpper = (64, 255, 255)

# change to a white range
pts = deque(maxlen=args["buffer"])

# get video stream if no video path was described
if not args.get("video", False):
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])

# pause for the file to warm up
time.sleep(2)

# Loop through video
while True:
	# get the current frame of the video
	frame = vs.read()

	# need to get frame[1] if using webcam
	frame = frame[1] if args.get("video", False) else frame

	# break if we have reached the end of a video
	if frame is None:
		break

	# resize, blur, and conver the frame
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# Create a mask for the color grean. Then use dilations and erosions to 
	# remove small blobs in the mask
	mask = cv2.inRange(hsv, colorLower, colorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# compute the contour of the green ball and draw it on the frame

	# initialize the current center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only draw contour of ball if we have a contour
	if len(cnts) > 0:
		# use the largest contour in the mask then compute the minimum enclosing
		# circle and centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size. Check if the contour
		# is large enough to be a ball
		if radius > 10:
			# draw circle around the ball
			cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)

			# draw centroid of the balll
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

			# Update the points queue
			pts.appendleft(center)

	# Draw the contrail of the balll
	for i in range(1, len(pts)):
		# if one of the tracked points is none, ignore
		if pts[i-1] is None or pts[i] is None:
			continue

		# compute thickness of the line and draw connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 2.5)
		cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), thickness)

	# Show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# break if 'q' key is pressed
	if key == ord("q"):
		break

# stop camera stream if necessary
if not args.get("video", False):
	vs.stop()

# release the camera if a video was used
else:
	vs.release()

# Close windows
cv2.destroyAllWindows()


