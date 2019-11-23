import numpy as np
from numpy.polynomial.polynomial import *

class RadialPolynomials:
	"""
	Class for calculating the Radial polynomials used in Zernike moment calculations.
	"""

	def __init__(self, maxP):
		self.maxP = maxP
		self._calculateRadialPolynomials()

	def _calculateRadialPolynomials(self):
		"""
		Create and store the polynomial objects
		for p = 0..maxP, q =  0..p,
		where p - q is even
		"""
		self.R = []
		for p in range(0,self.maxP + 1):
			self.R.append([])
			for q in range(0,p+1):
				self.R[p].append(np.array([]))

		for p in range(0,self.maxP + 1):
			for q in range(p,-1,-1):
				if ((p - q) % 2) == 1:
					self.R[p][q] = np.array([0])
					continue
				
				if p == q:
					self.R[p][q] = np.hstack((np.zeros(p), np.ones(1)))
					continue
				
				if p - q == 2:
					self.R[p][q] = polysub((q+2)*self.R[p][p], (q+1)*self.R[q][q])
					continue

				K1 = (p + q)*(p - q)*(p - 2)/2
				K2 = 2*p*(p - 1)*(p - 2)
				K3 = (-1)*q*q*(p - 1) - p*(p - 1)*(p - 2)
				K4 = (-1)*p*(p + q - 2)*(p - q - 2)/2

				r2 = np.array([0, 0, 1])
				self.R[p][q] = polyadd(polymul(polyadd(K2*r2, K3), self.R[p - 2][q]), K4*self.R[p-4][q]) / K1

	def value(self, p, q, r):
		"""
		Calculate the value of the polynomial R[p][q] at the given rs.
		(Also works with a single r value)
		"""
		if(p <= self.maxP and abs(q) <= p):
			return polyval(r, self.R[p][q])
		else:
			return 0


	def getPolynomial(self, p, q):
		"""
		Returns the representation of the polynomial R[p][q] in the form of its coefficitents.
		E.g. for the polynomial 1 + 2*x + 3*x^2 return [1,2,3]
		"""
		if(p <= self.maxP and abs(q) <= p):
			return self.R[p][abs(q)]
		else: 
			return np.array([0])