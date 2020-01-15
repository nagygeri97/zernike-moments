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

def calculateCentroid(img):
	m00 = 0
	m10 = 0
	m01 = 0
	(N, _, _) = img.shape
	for x in range(N):
		for y in range(N):
			s = sum([img[x,y,z] for z in range(3)])
			m10 += x*s
			m01 += y*s
			m00 += s
	m01 = int(round(float(m01) / m00))
	m10 = int(round(float(m10) / m00))
	return (m10, m01)