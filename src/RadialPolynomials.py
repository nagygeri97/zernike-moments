import numpy as np

class RadialPolynomials:
	"""
	Class for calculating the Radial polynomials used in Zernike moment calculations.
	"""

	def __init__(self, maxP):
		self.maxP = maxP
		self.values = np.empty([self.maxP + 1, self.maxP + 1])
	
	def calculateRadialPolynomials(self, r):
		"""
		Create and store the values of the polynomials
		for a single given r,
		for p = 0..maxP, q =  0..p,
		where p - q is even
		"""
		for p in range(0,self.maxP + 1):
			h = p*(p - 1)*(p - 2)
			K2 = 2*h
			for q in range(p,-1,-1):
				if ((p - q) % 2) == 1:
					self.values[p,q] = 0
					continue
				
				if p == q:
					self.values[p,q] = (r**p)
					continue
				
				if p - q == 2:
					self.values[p,q] = (q+2)*self.values[p,p] - (q+1)*self.values[q,q]
					continue

				K1 = (p + q)*(p - q)*(p - 2)/2
				K3 = (-1)*q*q*(p - 1) - h
				K4 = (-1)*p*(p + q - 2)*(p - q - 2)/2

				r2 = r**2
				self.values[p,q] = ((K2*r2 + K3)*self.values[p-2,q] + K4*self.values[p-4,q]) / K1
