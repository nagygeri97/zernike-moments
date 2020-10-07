import numpy as np
from numba import *
from PIL import Image

from ImageManipulation import *
from Transformations import *

class ZernikeMomentsMonochrome:
	"""
	Class for storing the Image, and the Transformation
	needed for calculating the Fourier moments (RHFMs).
	"""

	def __init__(self, img, colorIndex, maxP, maxQ, trans = None, sins = None, coss = None):
		# img not needed, trans contains the interpolated image
		self.colorIndex = colorIndex
		self.maxP = maxP
		self.maxQ = maxQ
		self.sins = sins
		self.coss = coss
		self.trans = trans # TODO: specify default
		self.calculateFourierMoments()
	
	def calculateFourierMoments(self):
		"""
		Calculate the Fourier moments (RHFMs) for
		p = 0..maxP and q = -maxQ..maxQ
		Only moments for q = 0..maxQ are stored, since the moment
		for -q is the conjugate of the moment for q
		"""
		# TODO: calculate sins, coss if not available

		self.Zre = np.zeros([self.maxP + 1, self.maxQ + 1])
		self.Zim = np.zeros([self.maxP + 1, self.maxQ + 1])

		# Calculate:
		zeros = np.zeros([self.maxP], dtype='float')
		calculate(self.trans.N, self.maxP, self.maxQ, self.colorIndex, self.trans.rs, self.trans.thetas, self.trans.img, self.trans.mu, self.sins, self.coss, self.Zre, self.Zim, zeros)

@jit(void(int64, int64, int64, int64, float64[:], float64[:], float64[:,:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:]), nopython=True)
def calculate(N, maxP, maxQ, colorIndex, rs, thetas, img, mu, sins, coss, Zre, Zim, zeros):
	for k in range(N):
		values = zeros.copy()
		calculateFourierKernel(rs[k], maxP, values)
		for j in range(N):
			for p in range(0, maxP + 1):
				for q in range(0, maxQ + 1):
					tmp = values[p] * img[k,j,colorIndex]
					Zre[p, q] += coss[j,q] * tmp
					Zim[p, q] += -sins[j,q] * tmp