import numpy as np
from PIL import Image

from RadialPolynomials import RadialPolynomials
from Transformations import *
from ZernikeMomentsMonochrome import *

class ZernikeMomentsColorRight:
	"""
	Class for calculating Zernike moments of an RGB image,
	using the ZernikeMomentsMonochrome class
	"""

	def __init__(self, img, N, maxP):
		self.img = img # img should contain RGB components
		self.N = N
		self.maxP = maxP
		self.trans = OldTransformation(N)
		self.Rs = RadialPolynomials(self.maxP)
		self._calculateZernikeMoments()

	def _calculateZernikeMoments(self):
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

		imgR = getColorComponent(self.img, 'R')
		imgG = getColorComponent(self.img, 'G')
		imgB = getColorComponent(self.img, 'B')
		ZfR = ZernikeMomentsMonochrome(imgR, self.N, self.maxP, self.trans, self.Rs, rs, thetas, sins, coss)
		ZfG = ZernikeMomentsMonochrome(imgG, self.N, self.maxP, self.trans, self.Rs, rs, thetas, sins, coss)
		ZfB = ZernikeMomentsMonochrome(imgB, self.N, self.maxP, self.trans, self.Rs, rs, thetas, sins, coss)
		self.ZfR = ZfR

		# the 3rd dim == 0 means q >= 0, ==1 means q < 0 
		self.Zre = np.zeros([self.maxP + 1, self.maxP + 1, 2])
		self.Zi  = np.zeros([self.maxP + 1, self.maxP + 1, 2])
		self.Zj  = np.zeros([self.maxP + 1, self.maxP + 1, 2])
		self.Zk  = np.zeros([self.maxP + 1, self.maxP + 1, 2])

		sqrt3inv = 1.0 / np.sqrt(3.0)

		for p in range(0, self.maxP + 1):
			for q in range(0, p + 1):
				if (p - q) % 2 != 0:
					continue
				# q >= 0
				self.Zre[p, q, 0] = - sqrt3inv * (ZfR.Zim[p, q] + ZfG.Zim[p, q] + ZfB.Zim[p, q])
				self.Zi[p, q, 0] = ZfR.Zre[p, q] + sqrt3inv * (ZfG.Zim[p, q] - ZfB.Zim[p, q])
				self.Zj[p, q, 0] = ZfG.Zre[p, q] + sqrt3inv * (ZfB.Zim[p, q] - ZfR.Zim[p, q])
				self.Zk[p, q, 0] = ZfB.Zre[p, q] + sqrt3inv * (ZfR.Zim[p, q] - ZfG.Zim[p, q])
				# q < 0
				if q == 0:
					continue
				self.Zre[p, q, 1] = sqrt3inv * (ZfR.Zim[p, q] + ZfG.Zim[p, q] + ZfB.Zim[p, q])
				self.Zi[p, q, 1] = ZfR.Zre[p, q] - sqrt3inv * (ZfG.Zim[p, q] - ZfB.Zim[p, q])
				self.Zj[p, q, 1] = ZfG.Zre[p, q] - sqrt3inv * (ZfB.Zim[p, q] - ZfR.Zim[p, q])
				self.Zk[p, q, 1] = ZfB.Zre[p, q] - sqrt3inv * (ZfR.Zim[p, q] - ZfG.Zim[p, q])
	
	def reconstructImageArray(self):
		errorNum = 0
		errorDen = 0
		imageArray = np.zeros((self.N, self.N, 3), dtype='uint8')
		sqrt3inv = 1.0 / np.sqrt(3.0)

		for x in range(self.N):
			for y in range(self.N):
				self.Rs.calculateRadialPolynomials(self.rs[x,y])
				value = [0, 0, 0] # RGB
				for p in range(0, self.maxP + 1):
					for q in range(0, p + 1):
						if (p - q) % 2 != 0:
							continue
						tmp = sqrt3inv * self.sins[x,y,q]
						value[0] += self.Rs.values[p,q] * (tmp * (self.Zre[p,q,0] + self.Zj[p,q,0] - self.Zk[p,q,0]) + self.coss[x,y,q]*self.Zi[p,q,0])
						value[1] += self.Rs.values[p,q] * (tmp * (self.Zre[p,q,0] + self.Zk[p,q,0] - self.Zi[p,q,0]) + self.coss[x,y,q]*self.Zj[p,q,0])
						value[2] += self.Rs.values[p,q] * (tmp * (self.Zre[p,q,0] + self.Zi[p,q,0] - self.Zj[p,q,0]) + self.coss[x,y,q]*self.Zk[p,q,0])
						if q == 0:
							continue
						value[0] += self.Rs.values[p,q] * (- tmp * (self.Zre[p,q,1] + self.Zj[p,q,1] - self.Zk[p,q,1]) + self.coss[x,y,q]*self.Zi[p,q,1])
						value[1] += self.Rs.values[p,q] * (- tmp * (self.Zre[p,q,1] + self.Zk[p,q,1] - self.Zi[p,q,1]) + self.coss[x,y,q]*self.Zj[p,q,1])
						value[2] += self.Rs.values[p,q] * (- tmp * (self.Zre[p,q,1] + self.Zi[p,q,1] - self.Zj[p,q,1]) + self.coss[x,y,q]*self.Zk[p,q,1])
				# Question: use this?
				for i in range(3):
					if value[i] > 255:
						value[i] = 255
					elif value[i] < 0:
						value[i] = 0
				for i in range(3):
					imageArray[x,y,i] = int(round(value[i]))
					errorNum += abs(imageArray[x,y,i] - float(self.img[x,y,i]))**2
					errorDen += abs(float(self.img[x,y,i]))**2
				print(x, y)
		eps = float(errorNum) / float(errorDen)
		return (imageArray, eps)

	def reconstructImage(self, fileName):
		(imgArray, eps) = self.reconstructImageArray()
		img = Image.fromarray(imgArray)
		img.save(fileName, "BMP")
		print("Mean square error: ", eps)