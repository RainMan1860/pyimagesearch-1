from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np 
import argparse
import imutils
import cv2
import os

'''
Flatten out hte vector, we will have 32 x 32 x 3 = 3072 elements

Utilizing raw pixel intensities as inputs to ML algorithms tends to yield poor
results as even small changes in rotation, translation, viewpoint, scale,, etc.
can dramatically influence the image itself

However: CNNs obtain fantastic results using raw pixel intensities as input
'''
# define a method to convert input image into a feature vector
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

# get color histogram
def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract HSV color
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

	# normalize histogram
	cv2.normalize(hist, hist)

	return hist.flatten()

# command line arguments. Input dataset. number of neighbors. number of jobs
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, 
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

'''
There are ways to reduce complexity using kd-trees or FLANN or Annoy. However,
we use exhaustive nearest neighbor search for simplicity
'''

# get images
print("[INFO] describing images...")
imagePath = list(paths.list_images(args["dataset"]))

# Initialize raw pixel intensities, features, and labels matrices
rawImages = []
features = []
labels = []

# Update the intensity, feature, and label matrices
for (i, imagePath) in enumerate(imagePath):
	# load image and extract labels
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	# extract raw pixel intensities and histogram
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

	# update matrices
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

	# Show an update every 1000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePath)))

# Create numpy matrices
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

# display number of megabytes of memory the representations utilize
print("[INFO] pixels matrix: {:.2f}MB".format(rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))

'''
Split data, 3/4 training  and 1/4 testing. Also ensure that it is the same split
for raw intensities and features by using random_state
'''

# Split our data into training and testing. For raw intensities and features
(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels,
	test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features,
	labels, test_size=0.25, random_state=42)

# Apply the k-NN classifier for raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[Info] raw pixel accuracy: {:.2f}%".format(acc * 100))

# Apply the k-NN classifier for color histogram features
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[Info] raw pixel accuracy: {:.2f}%".format(acc * 100))