from bisect import bisect
from numba import *
import numpy as np

from ImageManipulation import *
from legendre.LegendreRoots import *

class LegendrePoints1():
	def __init__(self, img, N=None):
		self.n, _, _ = img.shape
		# self.N = int(np.floor(float(self.n) / 4.0 * np.sqrt(np.pi)))
		# Same number of points as in the inscribed circle of the image
		self.N = N if N is not None else int(np.floor(float(-1 + np.sqrt(1 + 4*self.n*self.n*np.pi)) / 8.0))
		# self.N = int(np.floor(float(-1 + np.sqrt(1 + 4*self.n*self.n*np.pi)) / 8.0))
		self.rs, self.thetas, self.mu = getPoints(self.N)

class LegendreTransformation1():
	"""
	About the same number of points as in circle inscribed in the original image
	Bilinear interpolation is used to get the pixel values at the points
	"""
	def __init__(self, img, maxP=None):
		"""
		img should be centroid translated!!!
		"""
		points = LegendrePoints1(img) if maxP is None else LegendrePoints1(img, maxP + 1) # N > maxP is needed for discrete orthogonality
		self.rs = points.rs
		self.thetas = points.thetas
		self.mu = points.mu
		self.n, _, _ = img.shape
		self.N = points.N

		self.img = np.zeros((self.N, 4*self.N + 1, 3)) # RGB values for each (r,theta)
		
		# Calculate RGB values
		interpolate(self.N, 4*self.N + 1, self.n, img, self.img, self.rs, self.thetas)
	
	def lam(self, p):
		return (p + 1)

class LegendrePoints2():
	def __init__(self, N=10):
		self.N = int(N)
		self.rs, self.thetas, self.mu = getPoints(self.N)

class LegendreTransformation2():
	def __init__(self, img):
		"""
		If maxP is provided, use as many points as needed for discrete orthogonality
		Otherwise use N=10 
		"""
		points = LegendrePoints2()
		self.rs = points.rs
		self.thetas = points.thetas
		self.mu = points.mu
		self.n, _, _ = img.shape
		self.N = points.N

		self.img = np.zeros((self.N, 4*self.N + 1, 3)) # RGB values for each (r,theta)
		self.counts = np.zeros((self.N, 4*self.N + 1))

		self.c1 = 2.0 / float(self.n - 1)
		for x in range(self.n):
			for y in range(self.n):
				s1 = self.c1*x - 1 
				s2 = self.c1*y - 1
				r = np.sqrt(s1**2 + s2**2)
				theta = np.arctan2(s2, s1)
				if(theta < 0):
					theta = 2*np.pi + theta

				if r >= 1:
					continue
				
				i = bisect(self.rs, r) # gives first index where rs[i] > r
				j = bisect(self.thetas, theta)
				
				dist = 2
				mini = 0
				minj = 0
				if i < len(self.rs) and j < len(self.thetas):
					dist_ = polarDist((self.rs[i], self.thetas[j]),(r, theta))
					if dist_ < dist:
						dist = dist_
						mini = i
						minj = j
				if i < len(self.rs) and j > 0:
					dist_ = polarDist((self.rs[i], self.thetas[j - 1]),(r, theta))
					if dist_ < dist:
						dist = dist_
						mini = i
						minj = j - 1
				if i > 0 and j < len(self.thetas):
					dist_ = polarDist((self.rs[i - 1], self.thetas[j]),(r, theta))
					if dist_ < dist:
						dist = dist_
						mini = i - 1
						minj = j
				if i > 0 and j > 0:
					dist_ = polarDist((self.rs[i - 1], self.thetas[j - 1]),(r, theta))
					if dist_ < dist:
						dist = dist_
						mini = i - 1
						minj = j - 1
				
				for i in range(3):
					self.img[mini, minj, i] += img[x,y, i]
				self.counts[mini, minj] += 1

		for k in range(self.N):
			for j in range(4 * self.N + 1):
				for i in range(3):
					if self.counts[k, j] > 0:
						self.img[k, j, i] = float(self.img[k, j, i])/self.counts[k, j]

	def lam(self, p):
		return (p + 1)

def LegendreTransformationDiscOrth(maxP):
	class LegendreTransformationDiscOrthWrapper(LegendreTransformation1):
		def __init__(self, img):
			LegendreTransformation1.__init__(self, img, maxP)
	
	return LegendreTransformationDiscOrthWrapper

def polarDist(p1, p2):
	(r1, t1) = p1
	(r2, t2) = p2
	d = np.sqrt(r1*r1 + r2*r2 - r1*r2*np.cos(t2 - t1))
	return d

def getPoints(N):
	rs = np.zeros(N)
	thetas = np.zeros(4*N + 1)
	mu = np.zeros(N)

	lambdas = calculateAllLegendreRoots(N)

	# Calculate rs
	for k in range(N):
		rs[k] = np.sqrt(float(1 + lambdas[k])/2.0)
	
	# Calculate thetas
	for j in range(4*N + 1):
		thetas[j] = float(2*np.pi*(j+1)) / (4*N + 1)

	# Calculate mu
	# using Gaussian quadrature formula (https://www.encyclopediaofmath.org/index.php/Gauss_quadrature_formula)
	for k in range(N):
		A = 2.0 / ((1 - lambdas[k]**2) * legendreDerValue(N, lambdas[k])**2)
		mu[k] = A / (2.0 * (4 * N + 1))

	return (rs, thetas, mu)