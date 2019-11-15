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