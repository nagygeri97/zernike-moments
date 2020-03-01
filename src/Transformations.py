import numpy as np

from Utility import *

class OldTransformation:
	"""
	Transform square inside circle
	([])
	"""
	def __init__(self, N, img): # img is unused
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

class OldTransformation2:
	"""
	Transform circle inside square
	[()]
	"""
	def __init__(self, N, img): # img is unused
		self.N = N
		self.c1 = 2 / (N - 1)
		self.c2 = -1

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
		return (p + 1)/float((self.N - 1)**2)

class EqualRadsTransformation:
	"""
	Pixels on the same circumference in the original image get equal radii
	The angle is the same as in OldTransformation
	"""
	def __init__(self, N, img): # img is unused
		self.N = N
		self.c1 = np.sqrt(2) / (N - 1)
		self.c2 = -1 / np.sqrt(2)

	def getPolarCoords(self, x, y):
		sx = self.c1*x + self.c2
		sy = self.c1*y + self.c2
		m = max(abs(sx), abs(sy))
		r = np.sqrt(m**2 + m**2)
		theta = np.arctan2(sx, sy)
		r *= 0.9
		return (r,theta)

	def lam(self, p):
		# return 2*(p + 1)/(np.pi * (self.N - 1)**2)
		return 2*(p + 1)/(0.9**2 * np.pi * (self.N - 1)**2)

class ReverseTransformation:
	def __init__(self, N, img): # img is unused
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

class CentroidTransformation:
	"""
	Same as OldTransformation, except the image is first translated so the centroid is in the middle of the image
	NOTE: getPolarCoords may return r > 1, handle with care !!!
	"""
	def __init__(self, N, img):
		print("Centroid Transformation is being used, be careful, it may cause problems!")
		self.N = N
		self.c1 = np.sqrt(2) / (N - 1)
		self.c2 = -1 / np.sqrt(2)
		self.cx, self.cy = calculateCentroid(img)

	def getPolarCoords(self, x, y):
		"""
		Return the polar coordinates corresponding to the cartesian (x,y) coordinates.
		The returned value is in the form (r, theta)
		NOTE: May return r > 1, handle with care !!!
		"""
		s1 = self.c1*(x - self.cx + self.N/2) + self.c2
		s2 = self.c1*(y - self.cy + self.N/2) + self.c2
		r = np.sqrt(s1**2 + s2**2)
		theta = np.arctan2(s2, s1)
		return (r, theta)

	def lam(self, p):
		return 2*(p + 1)/(np.pi * (self.N - 1)**2)
