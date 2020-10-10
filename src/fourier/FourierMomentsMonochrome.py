import numpy as np
from numba import *
from PIL import Image

from ImageManipulation import *
from Transformations import *
from RadialPolynomials import *
from fourier.TransformationsFourier import *

class FourierMomentsMonochrome:
	"""
	Class for storing the Image, and the Transformation
	needed for calculating the Fourier moments (RHFMs).
	"""

	def __init__(self, img, colorIndex, maxP, maxQ, trans = None, sins = None, coss = None):
		# img only needed if trans is missing (and reconstruction), otherwise trans contains the interpolated image
		self.img = img

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
		if self.sins is None:
			if self.trans is None:
				self.trans = FourierTransformation(self.img, None)
			self.sins = np.empty([self.trans.N, self.maxQ + 1])
			self.coss = np.empty([self.trans.N, self.maxQ + 1])

			prepare(self.trans.N, self.maxQ, self.trans.thetas, self.sins, self.coss)

		self.Zre = np.zeros([self.maxP + 1, self.maxQ + 1])
		self.Zim = np.zeros([self.maxP + 1, self.maxQ + 1])

		# Calculate:
		zeros = np.zeros([self.maxP + 1], dtype='float')
		calculate(self.trans.N, self.maxP, self.maxQ, self.colorIndex, self.trans.rs, self.trans.thetas, self.trans.img, self.sins, self.coss, self.Zre, self.Zim, zeros)

		for p in range(self.maxP + 1):
			for q in range(self.maxQ + 1):
				self.Zre[p,q] *= self.trans.lam
				self.Zim[p,q] *= self.trans.lam

		# print moments:
		for p in range(self.maxP + 1):
			for q in range(self.maxQ + 1):
				print("M_{},{} = {} + {}i".format(p,q,self.Zre[p,q], self.Zim[p,q]))

	def reconstructImage(self, fileName, size):
		print("Color image reconstruction started.")
		errorNum = 0
		errorDen = 0
		imgArray = np.zeros((size, size, 3), dtype='uint8')
		image1d = np.zeros((size, size), dtype='uint8')
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

		reconstructImageArray(size, self.maxP, self.maxQ, rs, sins, coss, self.Zre, self.Zim, image1d, zeros)
		for x in range(size):
			for y in range(size):
				if rs[x,y] > 1: # handling r > 1, do not calculate with those values
					continue
				for i in range(3):
					imgArray[x,y,i] = image1d[x,y]
				errorNum += abs(imgArray[x,y,0] - float(self.img[x,y,0]))**2
				errorDen += abs(float(self.img[x,y,0]))**2
		img = Image.fromarray(imgArray)
		img.save(fileName)

		eps = float(errorNum) / float(errorDen)
		print("Mean square error: ", eps)
		return eps


# @jit(void(int64, int64, int64, int64, float64[:], float64[:], float64[:,:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:]), nopython=True)
def calculate(N, maxP, maxQ, colorIndex, rs, thetas, img, sins, coss, Zre, Zim, zeros):
	for k in range(N):
		values = zeros.copy()
		calculateFourierKernel(rs[k], maxP + 1, values)
		# print("R = " + str(rs[k]))
		# for i in range(len(values)):
		# 	print("{}. = {}".format(i,values[i]))
		for j in range(N):
			for p in range(0, maxP + 1):
				for q in range(0, maxQ + 1):
					tmp = values[p] * img[k,j,colorIndex]
					Zre[p, q] += coss[j,q] * tmp
					Zim[p, q] += -sins[j,q] * tmp

# @jit(void(int64, int64, int64, float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], uint8[:,:], float64[:]), nopython=False)
def reconstructImageArray(N, maxP, maxQ, rs, sins, coss, Zre, Zim, imageArray, zeros):
	for x in range(N):
		for y in range(N):
			if rs[x,y] > 1: # handling r > 1, do not calculate with those values
				continue
			values = zeros.copy()
			calculateFourierKernel(rs[x,y], maxP, values)
			value = 0 # Monochrome
			for p in range(0, maxP + 1):
				value += values[p] * (Zre[p,0] * coss[x,y,0] - Zim[p,0] * sins[x,y,0])
				for q in range(1, maxQ + 1):
					value += 2 * values[p] * (Zre[p,q] * coss[x,y,q] - Zim[p,q] * sins[x,y,q])

			# print("x {}, y {}, value {}".format(x,y,value))
			if value > 255:
				value = 255
			elif value < 0:
				value = 0
			imageArray[x,y] = int(round(value))

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