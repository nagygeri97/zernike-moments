import numpy as np
from numba import *
from PIL import Image

from RadialPolynomials import *
from Transformations import *
from legendre.TransformationsLegendre import *
from legendre.ZernikeMomentsMonochromeLegendre import *

class ZernikeMomentsColorRightLegendre:
	"""
	Class for calculating Zernike moments of an RGB image,
	using the ZernikeMomentsMonochrome class
	"""

	def __init__(self, img, maxP, verbose = False):
		self.img = img # img should contain RGB components
		self.maxP = maxP
		self.verbose = verbose

		self.trans = LegendreTransformation1(img)

		self.calculateZernikeMoments()

	def calculateZernikeMoments(self):
		if self.verbose:
			print("Zernike moment calculation started.")

		self.sins = np.empty([4*self.trans.N + 1, self.maxP + 1])
		self.coss = np.empty([4*self.trans.N + 1, self.maxP + 1])

		prepare(self.trans.N, self.maxP, self.trans.thetas, self.sins, self.coss)

		ZfR = ZernikeMomentsMonochromeLegendre(0, self.maxP, self.trans, self.sins, self.coss)
		if self.verbose:
			print("R done")
		ZfG = ZernikeMomentsMonochromeLegendre(1, self.maxP, self.trans, self.sins, self.coss)
		if self.verbose:
			print("G done")
		ZfB = ZernikeMomentsMonochromeLegendre(2, self.maxP, self.trans, self.sins, self.coss)
		if self.verbose:
			print("B done")

		self.Zre = np.zeros([self.maxP + 1, self.maxP + 1])
		self.Zi  = np.zeros([self.maxP + 1, self.maxP + 1])
		self.Zj  = np.zeros([self.maxP + 1, self.maxP + 1])
		self.Zk  = np.zeros([self.maxP + 1, self.maxP + 1])

		sqrt3inv = 1.0 / np.sqrt(3.0)

		for p in range(0, self.maxP + 1):
			# if p % 2 == 0: # logically it is needed, but bc of implementation, it can be removed
			self.Zre[p, 0] = - sqrt3inv * (ZfR.Zim[p, 0] + ZfG.Zim[p, 0] + ZfB.Zim[p, 0])
			self.Zi[p, 0] = ZfR.Zre[p, 0] + sqrt3inv * (ZfG.Zim[p, 0] - ZfB.Zim[p, 0])
			self.Zj[p, 0] = ZfG.Zre[p, 0] + sqrt3inv * (ZfB.Zim[p, 0] - ZfR.Zim[p, 0])
			self.Zk[p, 0] = ZfB.Zre[p, 0] + sqrt3inv * (ZfR.Zim[p, 0] - ZfG.Zim[p, 0])
			for q in range(p  % (-2) + 2, p + 1, 2):
				self.Zre[p, q] = - sqrt3inv * (ZfR.Zim[p, q] + ZfG.Zim[p, q] + ZfB.Zim[p, q])
				self.Zi[p, q] = ZfR.Zre[p, q] + sqrt3inv * (ZfG.Zim[p, q] - ZfB.Zim[p, q])
				self.Zj[p, q] = ZfG.Zre[p, q] + sqrt3inv * (ZfB.Zim[p, q] - ZfR.Zim[p, q])
				self.Zk[p, q] = ZfB.Zre[p, q] + sqrt3inv * (ZfR.Zim[p, q] - ZfG.Zim[p, q])

				# Calculation for -q, not needed
				# self.Zre[p, q] = sqrt3inv * (ZfR.Zim[p, q] + ZfG.Zim[p, q] + ZfB.Zim[p, q])
				# self.Zi[p, q] = ZfR.Zre[p, q] - sqrt3inv * (ZfG.Zim[p, q] - ZfB.Zim[p, q])
				# self.Zj[p, q] = ZfG.Zre[p, q] - sqrt3inv * (ZfB.Zim[p, q] - ZfR.Zim[p, q])
				# self.Zk[p, q] = ZfB.Zre[p, q] - sqrt3inv * (ZfR.Zim[p, q] - ZfG.Zim[p, q])
		if self.verbose:
			print("Zernike moment calculation done.")

@jit(void(int64, int64, float64[:], float64[:,:], float64[:,:]), nopython=True)
def prepare(N, maxP, thetas, sins, coss):
	for j in range(4*N + 1):
			for q in range(0, maxP + 1):
				sins[j,q] = np.sin(q*thetas[j])
				coss[j,q] = np.cos(q*thetas[j])
