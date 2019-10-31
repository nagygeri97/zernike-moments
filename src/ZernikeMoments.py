import numpy as np
import quaternion
from mpmath import *
from PIL import Image

from RadialPolynomialValues import RadialPolynomialValues
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
		self.Rs = RadialPolynomialValues(self.maxP)
		self._calculateZernikeMoments()

	def _calculateZernikeMoments(self):
		"""
		Calculate the Zernike moments for
		p = 0..maxP and q = -p..p, where p - q is even
		"""
		rs = []
		thetas = []
		sins = []
		coss = []
		for i in range(self.N):
			rs.append([])
			thetas.append([])
			sins.append([])
			coss.append([])
			for j in range(self.N):
				rs[i].append(0)
				thetas[i].append(0)
				sins[i].append([])
				coss[i].append([])
				for k in range(self.maxP + 1):
					sins[i][j].append(0)
					coss[i][j].append(0)

		for x in range(self.N):
			for y in range(self.N):
				r, theta = self.trans.getPolarCoords(x,y)
				if r > 1:
					r = mpf(1.0)
				rs[x][y] = r
				thetas[x][y] = theta
				for q in range(0, self.maxP + 1):
					sins[x][y][q] = mp.sin(q*theta)
					coss[x][y][q] = mp.cos(q*theta)
		
		self.rs = rs
		self.thetas = thetas
		self.sins = sins
		self.coss = coss

		self.Zre = []
		self.Zim = []

		# Initialize:
		for p in range(0, self.maxP + 1):
			self.Zre.append({})
			self.Zim.append({})
			for q in range(0, p + 1):
				self.Zre[p][q] = 0
				self.Zim[p][q] = 0
				self.Zre[p][-q] = 0
				self.Zim[p][-q] = 0

		# Calculate:
		for x in range(self.N):
			for y in range(self.N):
				self.Rs.calculateRadialPolynomials(rs[x][y])
				for p in range(0, self.maxP + 1):
					for q in range(0, p + 1):
						if (p - q) % 2 == 0:
							self.Zre[p][q] += self.Rs.values[p][q] * coss[x][y][q] * self.img[x, y]
							self.Zim[p][q] += self.Rs.values[p][q] * (-sins[x][y][q]) * self.img[x, y]
				print(x, y)
					
		# Scale:
		for p in range(0, self.maxP + 1):
			for q in range(0, p + 1):		
				l = self.trans.lam(p)
				self.Zre[p][q] *= l
				self.Zim[p][q] *= l
				self.Zre[p][-q] = self.Zre[p][q]
				self.Zim[p][-q] = self.Zim[p][q]
				print(p,q,self.Zre[p][q], self.Zim[p][q])

	def reconstructImage(self, filename):
		errorNum = 0
		errorDen = 0
		imgArray = np.zeros((self.N, self.N, 3), dtype='uint8')
		for x in range(self.N):
			for y in range(self.N):
				self.Rs.calculateRadialPolynomials(self.rs[x][y])
				c = 0
				for p in range(0, self.maxP + 1):
					for q in range(0, p + 1):
						if (p - q) % 2 != 0:
							continue
						rval = self.Rs.values[p][q]
						if(rval < -1 or rval > 1):
							print(x, y, p, q, self.rs[x][y], rval)
						tmp = rval * (self.Zre[p][q] * self.coss[x][y][q] -  self.Zim[p][q] * self.sins[x][y][q])
						c += 2 * tmp
				imgArray[x, y, 0] = int(round(c))
				errorNum += abs(int(round(c)) - self.img[x, y])**2
				errorDen += abs(self.img[x, y])**2
				imgArray[x,y, 1] = imgArray[x,y, 0]
				imgArray[x,y, 2] = imgArray[x,y, 0]
				print(x, y)
		img = Image.fromarray(imgArray)
		img.save(filename, "BMP")

		eps = mpf(errorNum) / mpf(errorDen)
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
