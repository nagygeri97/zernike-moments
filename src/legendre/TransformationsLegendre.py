from numba import *
import numpy as np

from ImageManipulation import *
from Utility import *
from legendre.LegendreRoots import *

class LegendrePoints1():
	def __init__(self, img):
		self.n, _, _ = img.shape
		# self.N = int(np.floor(float(self.n) / 4.0 * np.sqrt(np.pi)))
		self.N = int(np.floor(float(-1 + np.sqrt(1 + 4*self.n*self.n*np.pi)) / 8.0))
		self.rs, self.thetas, self.mu = getPoints(self.N)

class LegendreTransformation1():
	"""
	About the same number of points as in circle inscribed in the original image
	Bilinear interpolation is used to get the pixel values at the points
	"""
	def __init__(self, img, points=None):
		"""
		img should be centroid translated!!!
		"""
		if points is None:
			points = LegendrePoints1(img)
		self.rs = points.rs
		self.thetas = points.thetas
		self.mu = points.mu
		self.n, _, _ = img.shape
		self.N = points.N

		self.img = np.zeros((self.N, 4*self.N + 1, 3)) # RGB values for each (r,theta)
		
		# Calculate RGB values
		interpolate(self.N, self.n, img, self.img, self.rs, self.thetas)
	
	def lam(self, p):
		return (p + 1)
		

@jit(void(int64, int64, uint8[:,:,:], float64[:,:,:], float64[:], float64[:]), nopython=True)
def interpolate(N, n, oldImg, newImg, rs, thetas):
	for k in range(N):
		for j in range(4*N + 1):
			# Polar coord to cartesian
			xy = rs[k] * (np.exp(thetas[j] * 1j))
			x = xy.real
			y = xy.imag

			# Transform back to original image dimensions (so that the circle is inside the square)
			x += 1
			y += 1
			x *= (n - 1) / 2
			y *= (n - 1) / 2

			# Bilinear interpolation
			x1 = int(np.floor(x))
			y1 = int(np.floor(y))
			x2 = int(np.ceil(x))
			y2 = int(np.ceil(y))

			if x1 == x2:
				if y1 == y2:
					for i in range(3):
						newImg[k,j,i] = oldImg[x1,y1,i]
				else:
					for i in range(3):
						newImg[k,j,i] = int(round((y2 - y) * oldImg[x1,y2,i] + (y - y1) * oldImg[x1,y1,i]))
			elif y1 == y2:
				for i in range(3):
					newImg[k,j,i] = int(round((x2 - x) * oldImg[x2,y1,i] + (x - x1) * oldImg[x1,y1,i]))
			else:
				for i in range(3):
					newImg[k,j,i] = int(round(oldImg[x1,y1,i]*(x2 - x)*(y2 - y) +\
									oldImg[x2,y1,i]*(x - x1)*(y2 - y) +\
									oldImg[x1,y2,i]*(x2 - x)*(y - y1) +\
									oldImg[x2,y2,i]*(x - x1)*(y - y1)))
			
			for i in range(3):
				if newImg[k,j,i] > 255:
					newImg[k,j,i] = 255
				elif newImg[k,j,i] < 0:
					newImg[k,j,i] = 0


class LegendrePoints2():
	def __init__(self, N=10):
		self.N = N
		self.rs, self.thetas, self.mu = getPoints(self.N)

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