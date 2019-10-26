import numpy as np
import quaternion
from numpy.polynomial.polynomial import *

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
		self.Rs = RadialPolynomials(maxP)
		self.trans = OldTransformation(N)
		self._calculateZernikeMoments()

	def _calculateZernikeMoments(self):
		"""
		Calculate the Zernike moments for
		p = 0..maxP and q = -p..p, where p - q is even
		"""
		coords = []
		for x in range(self.N):
			coords.append([])
			for y in range(self.N):
				coords[x].append(self.trans.getPolarCoords(x,y))
		
		self.Z = []
		for p in range(0, self.maxP + 1):
			self.Z.append([])
			for q in range(-p, p + 1):
				qInd = q + p
				self.Z[p].append(0)
				if (p - abs(q)) % 2 == 0:
					for x in range(self.N):
						for y in range(self.N):
							r, theta = coords[x][y]
							self.Z[p][qInd] += self.Rs.values(p, abs(q), r) * np.exp(np.complex(0, -1 * q * theta)) * self.img[x,y]
					self.Z[p][qInd] *= self.trans.lam(p)
					print(p,q,self.Z[p][qInd])

		import pdb; pdb.set_trace()
		


def getColorComponent(img, color='R'):
	if color == 'R':
		index = 0
	elif color == 'G':
		index = 1
	else:
		index = 2
	(heigth, width, _) = img.shape
	monochromeImg = np.empty((heigth, width))
	for i in range(heigth):
		for j in range(width):
			monochromeImg[i,j] = img[i,j][index]

	return monochromeImg
