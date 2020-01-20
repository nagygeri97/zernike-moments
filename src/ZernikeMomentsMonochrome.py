import numpy as np
from numba import *
from PIL import Image

from RadialPolynomials import *
from Transformations import *

class ZernikeMomentsMonochrome:
	"""
	Class for storing the Image, the Transformation and the RadialPolynomials
	needed for calculating the Zernike moments.
	"""

	def __init__(self, img, N, maxP, transformation = None, rs = None, thetas = None, sins = None, coss = None):
		self.img = img # img should be a flat NxN array
		self.N = N
		self.maxP = maxP
		self.trans = transformation if transformation is not None else OldTransformation(N, img)
		self.rs = rs
		self.thetas = thetas
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
		if self.rs is None:
			self.rs = np.empty([self.N, self.N])
			self.thetas = np.empty([self.N, self.N])
			self.sins = np.empty([self.N, self.N, self.maxP + 1])
			self.coss = np.empty([self.N, self.N, self.maxP + 1])

			for x in range(self.N):
				for y in range(self.N):
					r, theta = self.trans.getPolarCoords(x,y) # maybe r > 1, handle later, at all times ignore if r > 1 
					self.rs[x,y] = r
					self.thetas[x,y] = theta
			prepare(self.N, self.maxP, self.thetas, self.sins, self.coss)

		self.Zre = np.zeros([self.maxP + 1, self.maxP + 1])
		self.Zim = np.zeros([self.maxP + 1, self.maxP + 1])

		# Calculate:
		zeros = np.zeros([self.maxP + 1, self.maxP + 1], dtype='float')
		calculate(self.N, self.maxP, self.rs, self.img, self.sins, self.coss, self.Zre, self.Zim, zeros)
					
		# Scale:
		for p in range(0, self.maxP + 1):
			for q in range(0, p + 1):
				l = self.trans.lam(p)
				self.Zre[p, q] *= l
				self.Zim[p, q] *= l

	def reconstructImage(self, fileName):
		errorNum = 0
		errorDen = 0
		imgArray = np.zeros((self.N, self.N, 3), dtype='uint8')
		image1d = np.zeros((self.N, self.N), dtype='uint8')
		zeros = np.zeros([self.maxP + 1, self.maxP + 1], dtype='float')
		reconstructImageArray(self.N, self.maxP, self.rs, self.sins, self.coss, self.Zre, self.Zim, image1d, zeros)
		for x in range(self.N):
			for y in range(self.N):
				for i in range(3):
					imgArray[x,y,i] = image1d[x,y]
				errorNum += abs(image1d[x,y] - self.img[x,y])**2
				errorDen += abs(self.img[x,y])**2
		img = Image.fromarray(imgArray)
		img.save(fileName, "BMP")

		eps = float(errorNum) / float(errorDen)
		print("Mean square error: ", eps)

@jit(void(int64, int64, float64[:,:], float64[:,:,:], float64[:,:,:]), nopython=True)
def prepare(N, maxP, thetas, sins, coss):
	for x in range(N):
		for y in range(N):
			for q in range(0, maxP + 1):
				sins[x,y,q] = np.sin(q*thetas[x,y])
				coss[x,y,q] = np.cos(q*thetas[x,y])

@jit(void(int64, int64, float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:,:]), nopython=True)
def calculate(N, maxP, rs, img, sins, coss, Zre, Zim, zeros):
	for x in range(N):
		for y in range(N):
			if rs[x,y] > 1: # handling r > 1, do not calculate with those values
				continue
			values = zeros.copy()
			calculateRadialPolynomials(rs[x,y], maxP, values)
			for p in range(0, maxP + 1):
				for q in range(p % 2, p + 1, 2):
					tmp = values[p,q] * img[x,y]
					Zre[p, q] += coss[x,y,q] * tmp
					Zim[p, q] += -sins[x,y,q] * tmp
					# Z[p, -q] == conjugate(Z[p, q])
			# print(x,y)

@jit(void(int64, int64, float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], uint8[:,:], float64[:,:]), nopython=True)
def reconstructImageArray(N, maxP, rs, sins, coss, Zre, Zim, imageArray, zeros):
	for x in range(N):
		for y in range(N):
			if rs[x,y] > 1: # handling r > 1, do not calculate with those values
				continue
			values = zeros.copy()
			calculateRadialPolynomials(rs[x,y], maxP, values)
			value = 0
			for p in range(0, maxP + 1):
				value += (p % 2 + 1) * values[p,0] * Zre[p,0]
				for q in range(p % (-2) + 2, p + 1, 2):
					# No need to calculate the imaginary part of the product, we know it is 0
					value += 2 * values[p,q] * (Zre[p,q] * coss[x,y,q] - Zim[p,q] * sins[x,y,q])

			if value > 255:
				value = 255
			elif value < 0:
				value = 0
			imageArray[x,y] = int(round(value))
			# print(x, y)


def getColorComponent(img, color='R'):
	if color == 'R':
		index = 0
	elif color == 'G':
		index = 1
	elif color == 'B':
		index = 2
	else:
		return
	(heigth, width, _) = img.shape
	monochromeImg = np.empty((heigth, width), dtype='double')
	for i in range(heigth):
		for j in range(width):
			monochromeImg[i,j] = img[i,j][index]

	return monochromeImg
