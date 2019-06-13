'''
This is the LeNet driver script

(1) Loading the MNIST dataset
(2) parititioning data into test and training
(3) loading and compiling the LeNet Architecture
(4) Training the network
(5) optinally saving the serialized weights to disk so it can be reused
(6) Displying visual examples of the network output to demonstrate that
our implementation is working correctly
'''

# import proper packages
from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from keras.datasets import mnist 
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K 
import numpy as np 
import argparse
import cv2

# Construct argument parse. Optional to save model, load model, or path to
# weights file
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str, 
	help="(optional) path to weights file")
args = vars(ap.parse_args())

'''
Default splits into 2/3 training data and 1/3 testing data
'''

# load the MNIST dataset and partition it
print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# reshape if using channels first ordering
if K.image_data_format() == "channels_first":
	trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
	testData = testData.reshape((testData.shape[0], 1, 28, 28))

# if we are using channels last order, design matrix to be
# samples x rows x columns x depth
else:
	trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
	testData = testData.reshape((testData.shape[0], 28, 28, 1))

# normalize data
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

'''
Convert our labels from integers to a one hot encoded vector

This is necessary for categorical cross-entropy loss
'''
# process labels so they can be used with the categorical cross-entropy loss
# function
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

'''
Training the network with Stochastic Gradient Descent. That's what SGD is
The learning rate of the SGD is 0.01

This is also where we build the model
'''

# intiialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

'''
If there is not a pre-existing model we train one

Additionally we print the resultls of the accuracy
'''

# train if we are not loading a pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=128, epochs=5, verbose=1)

	# show accuracy
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128,
		verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# save model if necessary
if args["save_model"] > 0:
	print("[INFO] dumping weights to file")
	model.save_weights(args["weights"], overwrite=True)

# radomly selects a few digits from our testing s et and passes them through
# the trained network
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):

	# classify the digit
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)

	# extract image
	if K.image_data_format() == "channels_first":
		image = (testData[i][0] * 255).astype("uint8")
	else:
		image = (testData[i] * 255).astype("uint8")

	# merge channels into one image, resize and show
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], 
		np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
