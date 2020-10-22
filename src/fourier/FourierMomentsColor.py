import numpy as np
from numba import *
from PIL import Image

from RadialPolynomials import *
from Transformations import *
from fourier.TransformationsFourier import *
from fourier.FourierMomentsMonochrome import *

class FourierMomentsColor:
	"""
	Class for calculating Fourier moments (AQRHFMs) of an RGB image,
	using the FourierMomentsMonochrome class
	"""

	def __init__(self, img, maxP, maxQ, transformationClass = None, verbose = False):
		self.img = img # img should contain RGB components
		self.maxP = maxP
		self.maxQ = maxQ
		self.verbose = verbose

		if transformationClass is None:
			transformationClass = FourierTransformation
		self.trans = transformationClass(img)

		self.calculateFourierMoments()

	def calculateFourierMoments(self):
		if self.verbose:
			print("Fourier moment calculation started.")

		self.sins = np.empty([self.trans.N, self.maxQ + 1])
		self.coss = np.empty([self.trans.N, self.maxQ + 1])

		prepare(self.trans.N, self.maxP, self.trans.thetas, self.sins, self.coss)

		ZfR = FourierMomentsMonochrome(0, self.maxP, self.maxQ, self.trans, self.sins, self.coss)
		if self.verbose:
			print("R done")
		ZfG = FourierMomentsMonochrome(1, self.maxP, self.maxQ, self.trans, self.sins, self.coss)
		if self.verbose:
			print("G done")
		ZfB = FourierMomentsMonochrome(2, self.maxP, self.maxQ, self.trans, self.sins, self.coss)
		if self.verbose:
			print("B done")
		
		self.ZfR = ZfR
		self.ZfG = ZfG
		self.ZfB = ZfB

		self.Zre = np.zeros([self.maxP + 1, self.maxQ + 1])
		self.Zi  = np.zeros([self.maxP + 1, self.maxQ + 1])
		self.Zj  = np.zeros([self.maxP + 1, self.maxQ + 1])
		self.Zk  = np.zeros([self.maxP + 1, self.maxQ + 1])

		sqrt3inv = 1.0 / np.sqrt(3.0)

		for p in range(self.maxP + 1):
			for q in range(self.maxQ + 1):
				self.Zre[p, q] = - sqrt3inv * (ZfR.Zim[p, q] + ZfG.Zim[p, q] + ZfB.Zim[p, q])
				self.Zi[p, q] = ZfR.Zre[p, q] + sqrt3inv * (ZfB.Zim[p, q] - ZfG.Zim[p, q])
				self.Zj[p, q] = ZfG.Zre[p, q] + sqrt3inv * (ZfR.Zim[p, q] - ZfB.Zim[p, q])
				self.Zk[p, q] = ZfB.Zre[p, q] + sqrt3inv * (ZfG.Zim[p, q] - ZfR.Zim[p, q])

				# Calculation for -q, not needed
				# self.Zre[p, q] = sqrt3inv * (ZfR.Zim[p, q] + ZfG.Zim[p, q] + ZfB.Zim[p, q])
				# self.Zi[p, q] = ZfR.Zre[p, q] - sqrt3inv * (ZfG.Zim[p, q] - ZfB.Zim[p, q])
				# self.Zj[p, q] = ZfG.Zre[p, q] - sqrt3inv * (ZfB.Zim[p, q] - ZfR.Zim[p, q])
				# self.Zk[p, q] = ZfB.Zre[p, q] - sqrt3inv * (ZfR.Zim[p, q] - ZfG.Zim[p, q])
		if self.verbose:
			print("Fourier moment calculation done.")

		# print moments:
		for p in range(self.maxP + 1):
			for q in range(self.maxQ + 1):
				print("M_{},{} = {} + {}i + {}j + {}k".format(p,q,self.Zre[p,q], self.Zi[p,q], self.Zj[p,q], self.Zk[p,q]))


	def reconstructImage(self, fileName, size):
		print("Color image reconstruction started.")
		errorNum = 0
		errorDen = 0
		imgArray = np.zeros((size, size, 3), dtype='uint8')
		zeros = np.zeros([self.maxP + 1], dtype='float')

		trans = OldTransformation2(size, self.img)
		rs = np.empty([size, size])
		thetas = np.empty([size, size])
		sins = np.empty([size, size, self.maxQ + 1])
		coss = np.empty([size, size, self.maxQ + 1])

		for x in range(size):
			for y in range(size):
				r, theta = trans.getPolarCoords(x,y) # maybe r > 1, handle later, at all times ignore if r > 1 
				rs[x,y] = r
				thetas[x,y] = theta
		prepareReconstruction(size, self.maxP, thetas, sins, coss)

		reconstructImageArray(size, self.maxP, self.maxQ, rs, sins, coss, self.Zre, self.Zi, self.Zj, self.Zk, self.ZfR.Zre, self.ZfG.Zre, self.ZfB.Zre, imgArray, zeros)
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

# @jit(void(int64, int64, float64[:], float64[:,:], float64[:,:]), nopython=True)
def prepare(N, maxQ, thetas, sins, coss):
	for j in range(N):
			for q in range(0, maxQ + 1):
				sins[j,q] = np.sin(q*thetas[j])
				coss[j,q] = np.cos(q*thetas[j])

# @jit(void(int64, int64, float64[:,:], float64[:,:,:], float64[:,:,:]), nopython=True)
def prepareReconstruction(N, maxP, thetas, sins, coss):
	for x in range(N):
		for y in range(N):
			for q in range(0, maxP + 1):
				sins[x,y,q] = np.sin(q*thetas[x,y])
				coss[x,y,q] = np.cos(q*thetas[x,y])

# @jit(void(int64, int64, int64, float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], uint8[:,:,:], float64[:]), nopython=False)
def reconstructImageArray(N, maxP, maxQ, rs, sins, coss, Zre, Zi, Zj, Zk, ZRRe, ZGRe, ZBRe, imageArray, zeros):
	sqrt3inv = 1.0 / np.sqrt(3.0)
	for x in range(N):
		for y in range(N):
			if rs[x,y] > 1: # handling r > 1, do not calculate with those values
				continue
			values = zeros.copy()
			calculateFourierKernel(rs[x,y], maxP, values)
			value = [0.0, 0.0, 0.0] # RGB
			for p in range(0, maxP + 1):
				tmp = sqrt3inv * sins[x,y,0]
				value[0] += values[0] * (tmp * (Zre[p,0] + Zj[p,0] - Zk[p,0]) + coss[x,y,0]*Zi[p,0])
				value[1] += values[0] * (tmp * (Zre[p,0] + Zk[p,0] - Zi[p,0]) + coss[x,y,0]*Zj[p,0])
				value[2] += values[0] * (tmp * (Zre[p,0] + Zi[p,0] - Zj[p,0]) + coss[x,y,0]*Zk[p,0])
				for q in range(1, maxQ + 1):
					tmp = sqrt3inv * sins[x,y,q]
					# New formula without explicitly calculating negative qs
					value[0] += 2 * values[q] * (tmp * (Zre[p,q] + (Zj[p,q] - ZGRe[p,q]) - (Zk[p,q] - ZBRe[p,q])) + coss[x,y,q] * ZRRe[p,q])
					value[1] += 2 * values[q] * (tmp * (Zre[p,q] + (Zk[p,q] - ZBRe[p,q]) - (Zi[p,q] - ZRRe[p,q])) + coss[x,y,q] * ZGRe[p,q])
					value[2] += 2 * values[q] * (tmp * (Zre[p,q] + (Zi[p,q] - ZRRe[p,q]) - (Zj[p,q] - ZGRe[p,q])) + coss[x,y,q] * ZBRe[p,q])
			print(value)
			for i in range(3):
				if value[i] > 255:
					value[i] = 255
				elif value[i] < 0:
					value[i] = 0
				imageArray[x,y,i] = int(round(value[i]))
