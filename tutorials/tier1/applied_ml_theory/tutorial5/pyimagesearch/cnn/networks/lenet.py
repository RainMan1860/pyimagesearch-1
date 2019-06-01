# import all of the required functions and classes
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K 

# define a clas sfor LeNet
class LeNet:
	@staticmethod
	def build(numChannels, imgRows, imgCols, numClasses, activation="relu",
		weightsPath=None):
		# This function requires height, width, and depth of input images

		# weights path can be used to load a pretrained model

		# Tensorflow default is channels last

		# initialize model and shape
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# update input shape if the channels are the first element in matrix
		if K.image_data_format == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# add layers to the model

		'''This is our first CONV -> RELU -> POOL layer sets

		learn 20 convolution filters where each size is 5x5.
		MaxPooling (2x2 sliding window that slides across the activation
		volume, taking the max operation of each region, while taking a step
		of 2 pixels in both the horizontal and vertical direction)
		'''
		
		model.add(Conv2D(20, 5, padding="same", input_shape=inputShape))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		'''
		Use 50 instead of 20 convolutional layers

		It is common to see the number of convolutional filters learned 
		increase in deeper layers of the network
		'''

		# Applpy second set
		model.add(Conv2D(50, 5, padding="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		'''
		Flatten into a sigle vector to allow us to apply dense/fully connected
		layers

		Also need a dense layer with the correct number of classes

		Softmax returnrs a probability, one for each of the 10 class labels
		'''

		# Add fully connected layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation(activation))

		# define the second Fully connected layer
		model.add(Dense(numClasses))

		# add soft-max classifier
		model.add(Activation("softmax"))

		# if a weights path is suppllied then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)

		return model