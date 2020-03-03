import numpy as np

from ImageManipulation import *
from Utility import *
from legendre.LegendreRoots import *

class LegendrePoints1():
	def __init__(self, img):
		self.n, _, _ = img.shape
		# self.N = int(np.floor(float(self.n) / 4.0 * np.sqrt(np.pi)))
		self.N = int(np.floor(float(-1 + np.sqrt(1 + 4*self.n*self.n*np.pi)) / 8.0))
		self.rs = np.zeros(self.N)
		self.thetas = np.zeros(4*self.N + 1)
		self.mu = np.zeros(self.N)

		self.lambdas = calculateAllLegendreRoots(self.N)

		# Calculate rs
		for k in range(self.N):
			self.rs[k] = np.sqrt(float(1 + self.lambdas[k])/2.0)
		
		# Calculate thetas
		for j in range(4*self.N + 1):
			self.thetas[j] = float(2*np.pi*(j+1)) / (4*self.N + 1)

		# Calculate mu
		LN = [1]
		for k in range(self.N):
			LN = np.polymul(LN, [1, -1 * self.lambdas[k]])
		
		for k in range(self.N):
			lkN, _ = np.polydiv(LN, [1, -1 * self.lambdas[k]])
			d = np.polyval(lkN, self.lambdas[k])
			lkN, _ = np.polydiv(lkN, [d])
			F = np.polyint(lkN)
			Ak = np.polyval(F, 1) - np.polyval(F, -1)
			self.mu[k] = float(Ak) / float(2*(4*self.N + 1))

class LegendreTransformation1():
	"""
	About the same number of points as in circle inscribed in the original image
	Bilinear interpolation is used to get the pixel values at the points
	"""
	def __init__(self, img, points):
		"""
		img should be centroid translated!!!
		"""		
		self.rs = points.rs
		self.thetas = points.thetas
		self.mu = points.mu
		self.n, _, _ = img.shape
		self.N = points.N

		self.img = np.zeros((self.N, 4*self.N + 1, 3)) # RGB values for each (r,theta)
		
		# Calculate RGB values
		for k in range(self.N):
			for j in range(4*self.N + 1):
				# Polar coord to cartesian
				xy = self.rs[k] * (np.exp(self.thetas[j] * 1j))
				x = xy.real
				y = xy.imag

				# Transform back to original image dimensions (so that the circle is inside the square)
				x += 1
				y += 1
				x *= (self.n - 1) / 2
				y *= (self.n - 1) / 2

				# Bilinear interpolation
				x1 = int(np.floor(x))
				y1 = int(np.floor(y))
				x2 = int(np.ceil(x))
				y2 = int(np.ceil(y))

				if x1 == x2:
					if y1 == y2:
						for i in range(3):
							self.img[k,j,i] = img[x1,y1,i]
					else:
						for i in range(3):
							self.img[k,j,i] = int(round((y2 - y) * img[x1,y2,i] + (y - y1) * img[x1,y1,i]))
				elif y1 == y2:
					for i in range(3):
						self.img[k,j,i] = int(round((x2 - x) * img[x2,y1,i] + (x - x1) * img[x1,y1,i]))
				else:
					for i in range(3):
						X = np.array([x2 - x, x - x1])
						M = np.array([[img[x1,y1,i], img[x1,y2,i]],
									[img[x2,y1,i], img[x2,y2,i]]])
						Y = np.array([y2 - y, y - y1])
						self.img[k,j,i] = int(round(np.matmul(np.matmul(X, M), Y)))
				
				for i in range(3):
					if self.img[k,j,i] > 255:
						self.img[k,j,i] = 255
					elif self.img[k,j,i] < 0:
						self.img[k,j,i] = 0
	def lam(self, p):
		return (p + 1)/float(np.pi * self.N * (4 * self.N + 1))
				
		


