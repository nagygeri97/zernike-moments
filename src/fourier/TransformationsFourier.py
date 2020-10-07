from numba import *
import numpy as np

from ImageManipulation import *
from Utility import *

class FourierPoints():
	def __init__(self, N):
		# self.n, _, _ = img.shape
		self.N = N # Total of N^2 points
		self.rs = np.zeros([N], dtype='float')
		self.thetas = np.zeros([N], dtype='float')

		for i in range(self.N):
			self.rs[i] = float(i) / self.N
			self.thetas[i] = float(i)*np.pi / self.N

class FourierTransformation():
	def __init__(self, img, N):
		points = FourierPoints(N)
		self.rs = points.rs
		self.thetas = points.thetas
		self.n, _, _ = img.shape
		self.N = points.N

		# Interpolated image
		self.img = np.zeros([self.N, self.N, 3]) # RGB values for each (r,theta)

		# Calculating RGB values
		interpolate(self.N, self.N, self.n, img, self.img, self.rs, self.thetas)
