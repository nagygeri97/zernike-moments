import bisect
import numpy as np
import quaternion
from PIL import Image

from RadialPolynomials import RadialPolynomials
from Transformations import *

class ZernikeMomentsMonochrome:
	"""
	Class for storing the Image, the Transformation and the RadialPolynomials
	needed for calculating the Zernike moments.
	"""

	def __init__(self, img, N, maxP, color='R'):
		self.img = getColorComponent(img, color)
		self.N = N
		self.maxP = maxP
		self.trans = OldTransformation(N)
		self.Rs = RadialPolynomials(self.maxP)
		self._calculateZernikeMoments()

	def _calculateZernikeMoments(self):
		"""
		Calculate the Zernike moments for
		p = 0..maxP and q = -p..p, where p - q is even
		Only moments for q = 0..p are stored, since the moment
		for -q is the conjugate of the moment for q
		"""
		rs = np.empty([self.N, self.N])
		thetas = np.empty([self.N, self.N])
		sins = np.empty([self.N, self.N, self.maxP + 1])
		coss = np.empty([self.N, self.N, self.maxP + 1])

		for x in range(self.N):
			for y in range(self.N):
				r, theta = self.trans.getPolarCoords(x,y)
				if r > 1:
					r = 1.0
				rs[x,y] = r
				thetas[x,y] = theta
				for q in range(0, self.maxP + 1):
					sins[x,y,q] = np.sin(q*theta)
					coss[x,y,q] = np.cos(q*theta)

		self.rs = rs
		self.thetas = thetas
		self.sins = sins
		self.coss = coss

		self.Z = np.zeros([self.maxP + 1, self.maxP + 1], dtype='complex')

		# Calculate:
		for x in range(self.N):
			for y in range(self.N):
				self.Rs.calculateRadialPolynomials(self.rs[x,y])
				for p in range(0, self.maxP + 1):
					for q in range(0, p + 1):
						if (p - q) % 2 != 0:
							continue
						rval = self.Rs.values[p,q]
						self.Z[p, q] += complex(coss[x,y,q], -sins[x,y,q]) * rval * self.img[x,y]
				print(x,y)
					
		# Scale:
		for p in range(0, self.maxP + 1):
			for q in range(0, p + 1):
				l = self.trans.lam(p)
				self.Z[p, q] *= l

	def reconstructImage(self, filename):
		errorNum = 0
		errorDen = 0
		imgArray = np.zeros((self.N, self.N, 3), dtype='uint8')
		for x in range(self.N):
			for y in range(self.N):
				self.Rs.calculateRadialPolynomials(self.rs[x,y])
				value = 0
				for p in range(0, self.maxP + 1):
					if p % 2 == 0:
						rval = self.Rs.values[p,0]
						tmp = rval * self.Z[p,0].real
						value += tmp
					for q in range(1, p + 1):
						if (p - q) % 2 != 0:
							continue
						rval = self.Rs.values[p,q]
						tmp = rval * (self.Z[p,q].real * self.coss[x,y,q] - self.Z[p,q].imag * self.sins[x,y,q])
						value += 2*tmp
				if value > 255:
					value = 255
				elif value < 0:
					value = 0
				imgArray[x,y,0] = int(round(value))
				errorNum += abs(int(round(value)) - self.img[x,y])**2
				errorDen += abs(self.img[x, y])**2
				imgArray[x,y,1] = imgArray[x,y,0]
				imgArray[x,y,2] = imgArray[x,y,0]
				print(x, y)
		img = Image.fromarray(imgArray)
		img.save(filename, "BMP")

		eps = float(errorNum) / float(errorDen)
		print("Mean square error =", eps)


def getColorComponent(img, color='R'):
	if color == 'R':
		index = 0
	elif color == 'G':
		index = 1
	else:
		index = 2
	(heigth, width, _) = img.shape
	monochromeImg = np.empty((heigth, width), dtype='double')
	for i in range(heigth):
		for j in range(width):
			monochromeImg[i,j] = img[i,j][index]

	return monochromeImg
