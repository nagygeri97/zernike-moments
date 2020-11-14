from numba import *
import numpy as np

from ImageManipulation import *

class FourierPoints():
	def __init__(self, N):
		# self.n, _, _ = img.shape
		self.N = N # Total of N^2 points
		self.rs = np.zeros([N], dtype='float')
		self.thetas = np.zeros([N], dtype='float')

		for i in range(self.N):
			self.rs[i] = float(i) / self.N
			self.thetas[i] = float(i)*2*np.pi / self.N

class FourierTransformationInterpolation():
	def __init__(self, img, N = None):
		self.n, _, _ = img.shape
		if N is not None:
			N = max(self.n, N)
		else:
			N = self.n
		points = FourierPoints(N)
		self.rs = points.rs
		self.thetas = points.thetas
		self.N = points.N
		self.lam = 1 / float(self.N * self.N)

		# Interpolated image
		self.img = np.zeros([self.N, self.N, 3]) # RGB values for each (r,theta)

		# Calculating RGB values
		interpolate(self.N, self.N, self.n, img, self.img, self.rs, self.thetas)

def FourierTransformationInterpolationDiscOrth(maxP):
	class FourierTransformationInterpolationDiscOrthWrapper(FourierTransformationInterpolation):
		def __init__(self, img):
			FourierTransformationInterpolation.__init__(self, img, 2*maxP)
	
	return FourierTransformationInterpolationDiscOrthWrapper

class FourierTransformationOriginal():
	def __init__(self, img, N = None):
		self.n, _, _ = img.shape
		if N is None:
			N = self.n
		points = FourierPoints(N)
		self.rs = points.rs
		self.thetas = points.thetas
		self.N = points.N
		self.lam = 1 / float(self.N * self.N)

		# Interpolated image
		self.img = np.zeros([self.N, self.N, 3]) # RGB values for each (r,theta)

		# Calculating RGB values
		halfN = float(self.N) / 2.0
		for i in range(N):
			for j in range(N):
				x = int(np.floor(self.rs[i] * halfN * np.cos(self.thetas[j])) + halfN)
				y = int(np.floor(self.rs[i] * halfN * np.sin(self.thetas[j])) + halfN)
				for color in range(3):
					self.img[i,j,color] = img[x,y,color]