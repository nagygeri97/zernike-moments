import numpy as np
from numba import *
from PIL import Image

from RadialPolynomials import *
from Transformations import *

class ZernikeMomentsMonochromeLegendre:
	"""
	Class for storing the Image, the Transformation and the RadialPolynomials
	needed for calculating the Zernike moments.
	"""
	def __init__(self, colorIndex, maxP, trans, sins, coss):
		self.colorIndex = colorIndex
		self.maxP = maxP
		self.trans = trans
		self.sins = sins
		self.coss = coss
		self.calculateZernikeMoments()

	def calculateZernikeMoments(self):
		"""
		Calculate the Zernike moments for
		p = 0..maxP and q = -p..p, where p - q is even
		Only moments for q = 0..p are stored, since the moment
		for -q is the conjugate of the moment for q
		"""
		self.Zre = np.zeros([self.maxP + 1, self.maxP + 1])
		self.Zim = np.zeros([self.maxP + 1, self.maxP + 1])

		# Calculate:
		zeros = np.zeros([self.maxP + 1, self.maxP + 1], dtype='float')
		calculate(self.trans.N, self.maxP, self.colorIndex, self.trans.rs, self.trans.thetas, self.trans.img, self.trans.mu, self.sins, self.coss, self.Zre, self.Zim, zeros)

		# Scale:
		for p in range(0, self.maxP + 1):
			for q in range(0, p + 1):
				l = self.trans.lam(p)
				self.Zre[p, q] *= l
				self.Zim[p, q] *= l

@jit(void(int64, int64, int64, float64[:], float64[:], float64[:,:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:]), nopython=True)
def calculate(N, maxP, colorIndex, rs, thetas, img, mu, sins, coss, Zre, Zim, zeros):
	for k in range(N):
		values = zeros.copy()
		calculateRadialPolynomials(rs[k], maxP, values)
		for j in range(4*N + 1):
			for p in range(0, maxP + 1):
				for q in range(p % 2, p + 1, 2):
					tmp = values[p,q] * img[k,j,colorIndex] * mu[k]
					Zre[p, q] += coss[j,q] * tmp
					Zim[p, q] += -sins[j,q] * tmp

