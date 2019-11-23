import numpy as np
from PIL import Image

from RadialPolynomials import RadialPolynomials
# from RadialPolynomialsSlow import RadialPolynomials
from Transformations import *

class ZernikeMomentsMonochrome:
	"""
	Class for storing the Image, the Transformation and the RadialPolynomials
	needed for calculating the Zernike moments.
	"""

	def __init__(self, img, N, maxP, transformation = None, radialPolynomials = None, rs = None, thetas = None, sins = None, coss = None):
		self.img = img # img should be a flat NxN array
		self.N = N
		self.maxP = maxP
		self.trans = transformation if transformation is not None else OldTransformation(N)
		self.Rs = radialPolynomials if radialPolynomials is not None else RadialPolynomials(self.maxP)
		self.rs = rs
		self.thetas = thetas
		self.sins = sins
		self.coss = coss
		self._calculateZernikeMoments()

	def _calculateZernikeMoments(self):
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
					r, theta = self.trans.getPolarCoords(x,y)
					if r > 1:
						r = 1.0
					self.rs[x,y] = r
					self.thetas[x,y] = theta
					for q in range(0, self.maxP + 1):
						self.sins[x,y,q] = np.sin(q*theta)
						self.coss[x,y,q] = np.cos(q*theta)

		self.Zre = np.zeros([self.maxP + 1, self.maxP + 1])
		self.Zim = np.zeros([self.maxP + 1, self.maxP + 1])

		# Calculate:
		for x in range(self.N):
			for y in range(self.N):
				self.Rs.calculateRadialPolynomials(self.rs[x,y])
				for p in range(0, self.maxP + 1):
					for q in range(p % 2, p + 1, 2):
						tmp = self.Rs.values[p,q] * self.img[x,y]
						self.Zre[p, q] += self.coss[x,y,q] * tmp
						self.Zim[p, q] += -self.sins[x,y,q] * tmp
						# Z[p, -q] == conjugate(Z[p, q])
				print(x,y)
					
		# Scale:
		for p in range(0, self.maxP + 1):
			for q in range(0, p + 1):
				l = self.trans.lam(p)
				self.Zre[p, q] *= l
				self.Zim[p, q] *= l

	def reconstructImageArray(self):
		errorNum = 0
		errorDen = 0
		imageArray = np.zeros((self.N, self.N), dtype='uint8')
		for x in range(self.N):
			for y in range(self.N):
				self.Rs.calculateRadialPolynomials(self.rs[x,y])
				value = 0
				for p in range(0, self.maxP + 1):
					value += (p % 2 + 1) * self.Rs.values[p,0] * self.Zre[p,0]
					for q in range(p % (-2) + 2, p + 1, 2):
						# No need to calculate the imaginary part of the product, we know it is 0
						value += 2 * self.Rs.values[p,q] * (self.Zre[p,q] * self.coss[x,y,q] - self.Zim[p,q] * self.sins[x,y,q])

				if value > 255:
					value = 255
				elif value < 0:
					value = 0
				imageArray[x,y] = int(round(value))
				errorNum += abs(imageArray[x,y] - self.img[x,y])**2
				errorDen += abs(self.img[x,y])**2
				print(x, y)

		eps = float(errorNum) / float(errorDen)
		return (imageArray, eps)

	def reconstructImage(self, fileName):
		imgArray = np.zeros((self.N, self.N, 3), dtype='uint8')
		(image1d, eps) = self.reconstructImageArray()
		for x in range(self.N):
			for y in range(self.N):
				for i in range(3):
					imgArray[x,y,i] = image1d[x,y]
		img = Image.fromarray(imgArray)
		img.save(fileName, "BMP")
		print("Mean square error: ", eps)


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
