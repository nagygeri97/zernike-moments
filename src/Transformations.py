import numpy as np

class OldTransformation:
	"""
	Functions calculating the transformation from cartesian to polar coordinates.
	"""
	def __init__(self, N):
		self.N = N
		self.c1 = np.sqrt(2) / (N - 1)
		self.c2 = -1 / np.sqrt(2)

	def getPolarCoords(self, x, y):
		"""
		Return the polar coordinates corresponding to the cartesian (x,y) coordinates.
		The returned value is in the form (r, theta).
		"""
		s1 = self.c1*x + self.c2
		s2 = self.c1*y + self.c2
		r = np.sqrt(s1**2 + s2**2)
		theta = np.arctan2(s2, s1)
		return (r, theta)

	def lam(self, p):
		return 2*(p + 1)/(np.pi * (self.N - 1)**2)

class EqualRadsTransformation:
	"""
	Pixels on the same circumference in the original image get equal radii
	The angle is the same as in OldTransformation
	"""
	def __init__(self, N):
		self.N = N
		self.c1 = np.sqrt(2) / (N)
		self.c2 = -1 / np.sqrt(2)

	def getPolarCoords(self, x, y):
		sx = self.c1*x + self.c2
		sy = self.c1*y + self.c2
		m = max(abs(sx), abs(sy))
		r = np.sqrt(m**2 + m**2)
		theta = np.arctan2(sx, sy)
		return (r,theta)

	def lam(self, p):
		return 2*(p + 1)/(np.pi * (self.N - 1)**2)

class ReverseTransformation:
	def __init__(self, N):
		self.N = N
		self.c1 = N/2
	
	def getCartesianCoords(self, r, theta):
		x = r*np.sin(theta)
		y = r*np.cos(theta)
		x *= self.c1
		y *= self.c1
		x += self.N/2
		y += self.N/2
		return (int(round(x)),int(round(y)))