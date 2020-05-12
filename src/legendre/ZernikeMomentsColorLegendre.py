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

	def __init__(self, img, maxP, transformationClass, verbose = False):
		self.img = img # img should contain RGB components
		self.maxP = maxP
		self.verbose = verbose

		self.trans = transformationClass(img)

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
		
		self.ZfR = ZfR
		self.ZfG = ZfG
		self.ZfB = ZfB

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

	def reconstructImage(self, fileName, size):
		print("Color image reconstruction started.")
		errorNum = 0
		errorDen = 0
		imgArray = np.zeros((size, size, 3), dtype='uint8')
		zeros = np.zeros([self.maxP + 1, self.maxP + 1], dtype='float')

		trans = OldTransformation2(size, self.img)
		rs = np.empty([size, size])
		thetas = np.empty([size, size])
		sins = np.empty([size, size, self.maxP + 1])
		coss = np.empty([size, size, self.maxP + 1])

		for x in range(size):
			for y in range(size):
				r, theta = trans.getPolarCoords(x,y) # maybe r > 1, handle later, at all times ignore if r > 1 
				rs[x,y] = r
				thetas[x,y] = theta
		prepareReconstruction(size, self.maxP, thetas, sins, coss)

		reconstructImageArray(size, self.maxP, rs, sins, coss, self.Zre, self.Zi, self.Zj, self.Zk, self.ZfR.Zre, self.ZfG.Zre, self.ZfB.Zre, imgArray, zeros)
		for x in range(size):
			for y in range(size):
				if rs[x,y] > 1: # handling r > 1, do not calculate with those values
					continue
				for i in range(3):
					errorNum += abs(imgArray[x,y,i] - float(self.img[x,y,i]))**2
					errorDen += abs(float(self.img[x,y,i]))**2
		img = Image.fromarray(imgArray)
		img.save(fileName)

		eps = float(errorNum) / float(errorDen)
		print("Mean square error: ", eps)
		return eps

@jit(void(int64, int64, float64[:], float64[:,:], float64[:,:]), nopython=True)
def prepare(N, maxP, thetas, sins, coss):
	for j in range(4*N + 1):
			for q in range(0, maxP + 1):
				sins[j,q] = np.sin(q*thetas[j])
				coss[j,q] = np.cos(q*thetas[j])

@jit(void(int64, int64, float64[:,:], float64[:,:,:], float64[:,:,:]), nopython=True)
def prepareReconstruction(N, maxP, thetas, sins, coss):
	for x in range(N):
		for y in range(N):
			for q in range(0, maxP + 1):
				sins[x,y,q] = np.sin(q*thetas[x,y])
				coss[x,y,q] = np.cos(q*thetas[x,y])

@jit(void(int64, int64, float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], uint8[:,:,:], float64[:,:]), nopython=False)
def reconstructImageArray(N, maxP, rs, sins, coss, Zre, Zi, Zj, Zk, ZRRe, ZGRe, ZBRe, imageArray, zeros):
	sqrt3inv = 1.0 / np.sqrt(3.0)
	for x in range(N):
		for y in range(N):
			if rs[x,y] > 1: # handling r > 1, do not calculate with those values
				continue
			values = zeros.copy()
			calculateRadialPolynomials(rs[x,y], maxP, values)
			value = [0.0, 0.0, 0.0] # RGB
			for p in range(0, maxP + 1):
				tmp = sqrt3inv * sins[x,y,0]
				value[0] += values[p,0] * (tmp * (Zre[p,0] + Zj[p,0] - Zk[p,0]) + coss[x,y,0]*Zi[p,0])
				value[1] += values[p,0] * (tmp * (Zre[p,0] + Zk[p,0] - Zi[p,0]) + coss[x,y,0]*Zj[p,0])
				value[2] += values[p,0] * (tmp * (Zre[p,0] + Zi[p,0] - Zj[p,0]) + coss[x,y,0]*Zk[p,0])
				for q in range(p % (-2) + 2, p + 1, 2):
					tmp = sqrt3inv * sins[x,y,q]
					# New formula without explicitly calculating negative qs
					value[0] += 2 * values[p,q] * (tmp * (Zre[p,q] + (Zj[p,q] - ZGRe[p,q]) - (Zk[p,q] - ZBRe[p,q])) + coss[x,y,q] * ZRRe[p,q])
					value[1] += 2 * values[p,q] * (tmp * (Zre[p,q] + (Zk[p,q] - ZBRe[p,q]) - (Zi[p,q] - ZRRe[p,q])) + coss[x,y,q] * ZGRe[p,q])
					value[2] += 2 * values[p,q] * (tmp * (Zre[p,q] + (Zi[p,q] - ZRRe[p,q]) - (Zj[p,q] - ZGRe[p,q])) + coss[x,y,q] * ZBRe[p,q])

					# Old formula for calculating with Z values for negative qs
					# value[0] += values[p,q] * (tmp * (Zre[p,q,0] + Zj[p,q,0] - Zk[p,q,0]) + coss[x,y,q]*Zi[p,q,0])
					# value[1] += values[p,q] * (tmp * (Zre[p,q,0] + Zk[p,q,0] - Zi[p,q,0]) + coss[x,y,q]*Zj[p,q,0])
					# value[2] += values[p,q] * (tmp * (Zre[p,q,0] + Zi[p,q,0] - Zj[p,q,0]) + coss[x,y,q]*Zk[p,q,0])

					# value[0] += values[p,q] * (- tmp * (Zre[p,q,1] + Zj[p,q,1] - Zk[p,q,1]) + coss[x,y,q]*Zi[p,q,1])
					# value[1] += values[p,q] * (- tmp * (Zre[p,q,1] + Zk[p,q,1] - Zi[p,q,1]) + coss[x,y,q]*Zj[p,q,1])
					# value[2] += values[p,q] * (- tmp * (Zre[p,q,1] + Zi[p,q,1] - Zj[p,q,1]) + coss[x,y,q]*Zk[p,q,1])

			for i in range(3):
				if value[i] > 255:
					value[i] = 255
				elif value[i] < 0:
					value[i] = 0
				imageArray[x,y,i] = int(round(value[i]))
