
#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):

	def __init__(self, x, y):
		"""
		Initialize an ImageGenerator instance.
		:param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
		:param y: A Numpy vector of labels. It has shape (num_of_samples, ).
		"""

		# TODO: Your ImageGenerator instance has to store the following information:
		# x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
		# is_vertical_flip, is_add_noise. By default, set boolean values to
		# False.
		# raise NotImplementedError
		self.x=x
		self.y=y
		self.num_of_samples=x.shape[0]
		self.height=x.shape[1]
		self.width=x.shape[2]
		self.xTranslatedPixels=None
		self.yTranslatedPixels=None
		self.degRotation=None
		self.is_horizontal_flip=False
		self.is_vertical_flip=False
		self.is_add_noise=False

	def next_batch_gen(self, batch_size, shuffle=True):
		"""
		A python generator function that yields a batch of data indefinitely.
		:param batch_size: The number of samples to return for each batch.
		:param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
						If False, the order or data samples stays the same.
		:return: A batch of data with size (batch_size, width, height, channels).
		"""

		# TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
		# 1. The generator should return batches endlessly.
		# 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
		# not be output.

		# One possible pseudo code for your reference:
		#######################################################################
		#   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
		#   while True:
		#	   if (batch_count < total number of batches possible):
		#		   batch_count = batch_count + 1
		#		   yield(next batch of x and y indicated by batch_count)
		#	   else:
		#		   shuffle(x)
		#		   reset batch_count
		# raise NotImplementedError

		x=self.x
		maxBatches=int(self.num_of_samples/batch_size)
		batchCount=0
		while True:
			 if(batchCount<maxBatches):
			 	batchCount+=1
			 	return x[(batchCount-1)*batch_size:batchCount*batch_size]
			 else:
			 	numpy.random.shuffle(x)
			 	batchCount=0









		#######################################################################
		#																	 #
		#																	 #
		#						 TODO: YOUR CODE HERE						#
		#																	 #
		#																	 #
		#######################################################################

	def show(self):
		"""
		Plot the top 16 images (index 0~15) of self.x for visualization.
		"""
		# raise NotImplementedError
		topX = self.x[:16]
		# Xrandom=Xrandom.reshape(16,32,32) 
		r = 4
		f, axarr = plt.subplots(r, r, figsize=(8,8))
		for i in range(r):
			for j in range(r):
				img = topX[r*i+j]
				axarr[i][j].imshow(img, cmap="gray")
		#######################################################################
		#																	 #
		#																	 #
		#						 TODO: YOUR CODE HERE						#
		#																	 #
		#																	 #
		#######################################################################

	def translate(self, shift_height, shift_width):
		"""
		Translate self.x by the values given in shift.
		:param shift_height: the number of pixels to shift along height direction. Can be negative.
		:param shift_width: the number of pixels to shift along width direction. Can be negative.
		:return:
		"""

		# TODO: Implement the translate function. Remember to record the value of the number of pixels translated.
		# Note: You may wonder what values to append to the edge after the translation. Here, use rolling instead. For
		# example, if you translate 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
		# right edge of the picture.
		# Hint: Numpy.roll
		# (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
		# raise NotImplementedError

		x=self.x
		# print(x.shape)
		x=np.roll(x,shift=(shift_height, shift_width), axis=(1,2))
		# x=np.roll(x, shift_width, axis=2)
		self.yTranlatedPixels=shift_height
		self.xTranlatedPixels=shift_width
		self.x=x
		# self.show()

		#######################################################################
		#																	 #
		#																	 #
		#						 TODO: YOUR CODE HERE						#
		#																	 #
		#																	 #
		#######################################################################

	def rotate(self, angle=0.0):
		"""
		Rotate self.x by the angles (in degree) given.
		:param angle: Rotation angle in degrees.

		- https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
		"""
		# TODO: Implement the rotate function. Remember to record the value of
		# rotation degree.
		# raise NotImplementedError

		x=self.x
		x=rotate(x,angle, axes=(1,2))
		self.degRotation=angle
		self.x=x
		# self.show()
		#######################################################################
		#																	 #
		#																	 #
		#						 TODO: YOUR CODE HERE						#
		#																	 #
		#																	 #
		#######################################################################

	def flip(self, mode='h'):
		"""
		Flip self.x according to the mode specified
		:param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
		"""
		# TODO: Implement the flip function. Remember to record the boolean values is_horizontal_flip and
		# is_vertical_flip.
		# raise NotImplementedError
		x=self.x
		if mode=="h":
			x=np.flip(x, axis=2)
			self.is_horizontal_flip=True
		elif mode=="v":
			x=np.flip(x, axis=1)
			self.is_vertical_flip=True
		elif mode=="hv":
			x=np.flip(x, axis=2)
			x=np.flip(x, axis=1)
			self.is_horizontal_flip=True
			self.is_vertical_flip=True
		else:
			return "Incorrect Argument"

		self.x=x
		# self.show()


		#######################################################################
		#																	 #
		#																	 #
		#						 TODO: YOUR CODE HERE						#
		#																	 #
		#																	 #
		#######################################################################

	def add_noise(self, portion, amplitude):

		"""
		Add random integer noise to self.x.
		:param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
						then 1000 samples will be noise-injected.
		:param amplitude: An integer scaling factor of the noise.
		"""
		# TODO: Implement the add_noise function. Remember to record the
		# boolean value is_add_noise. You can try uniform noise or Gaussian
		# noise or others ones that you think appropriate.
		# raise NotImplementedError


		self.is_add_noise=True

		x=self.x
		idx=np.random.choice(x.shape[0], size=int(self.x.shape[0]*portion), replace=False)
		noise=np.zeros(shape=x.shape)
		# print(noise.shape)
		for i in idx:
			noise[i]=np.random.uniform(size=(x.shape[1], x.shape[2], x.shape[3]))*amplitude
		# print(noise.shape)
		# print(noise[:10])
		# print(self.x[:10])


		self.x=x+noise
		# print(self.x[:10])
		# self.show()

        
        

        # for i in idx:
            # noise[]
	


		#######################################################################
		#																	 #
		#																	 #
		#						 TODO: YOUR CODE HERE						#
		#																	 #
		#																	 #
		#######################################################################
