import numpy as np
from numba import *
from PIL import Image

from RadialPolynomials import *
from Transformations import *
from fourier.TransformationsFourier import *
from fourier.FourierMomentsMonochrome import *
from fourier.FourierMomentsColor import *

class FourierMomentsInvariantColor:
	def __init__(self, img, maxP, transformationClass, noiseFun = None, centroidTranslate = True):
		self.img = img
		self.maxP = maxP

		# Add noise before/after centroidTranslation
		if noiseFun is not None:
			self.img = noiseFun(self.img)

		if centroidTranslate:
			self.img = centroidTranslationFloat(self.img)
		else:
			self.img = imageToFloat(self.img)

		# saveImgFromNpArray(img, "../original.png")
		# saveImgFromNpArray(self.img, "../test.png")

		self.FM = FourierMomentsColor(self.img, self.maxP, self.maxP, transformationClass)
		
		# Calculate modulus of Fourier moments, as they are invariant to image rotation.
		self.FMs = np.zeros([self.maxP + 1, self.maxP + 1])
		for p in range(self.maxP + 1):
			for q in range(self.maxP + 1):
				self.FMs[p,q] = np.sqrt(self.FM.Zre[p,q]**2 + self.FM.Zi[p,q]**2 + self.FM.Zj[p,q]**2 + self.FM.Zk[p,q]**2)
	

class FourierMomentsInvariantInterpolation(FourierMomentsInvariantColor):
	def __init__(self, img, maxP, noiseFun = None):
		super().__init__(img, maxP, FourierTransformationInterpolation, noiseFun)

class FourierMomentsInvariantOriginal(FourierMomentsInvariantColor):
	def __init__(self, img, maxP, noiseFun = None):
		super().__init__(img, maxP, FourierTransformationOriginal, noiseFun)

class FourierMomentsRotationInvariantInterpolation(FourierMomentsInvariantColor):
	def __init__(self, img, maxP, noiseFun = None):
		super().__init__(img, maxP, FourierTransformationInterpolation, noiseFun, False)

class FourierMomentsRotationInvariantOriginal(FourierMomentsInvariantColor):
	def __init__(self, img, maxP, noiseFun = None):
		super().__init__(img, maxP, FourierTransformationOriginal, noiseFun, False)