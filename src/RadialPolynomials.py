import numpy as np
# from numpy.polynomial.polynomial import *

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
				self.R[p].append([])

		for p in range(0,self.maxP + 1):
			for q in range(p,-1,-1):
				if ((p - q) % 2) == 1:
					self.R[p][q] = [0]
					continue
				
				if p == q:
					for i in range(p):
						self.R[p][q].append(0)
					self.R[p][q].append(1)
					continue
				
				if p - q == 2:
					self.R[p][q] = polysub(constmul(q+2, self.R[p][p]), constmul(q+1, self.R[q][q]))
					continue

				K1 = (p + q)*(p - q)*(p - 2)//2
				K2 = 2*p*(p - 1)*(p - 2)
				K3 = (-1)*q*q*(p - 1) - p*(p - 1)*(p - 2)
				K4 = (-1)*p*(p + q - 2)*(p - q - 2)//2

				r2 = [0, 0, 1]
				self.R[p][q] = constdiv(polyadd(polymul(polyadd(constmul(K2, r2), [K3]), self.R[p-2][q]), constmul(K4, self.R[p-4][q])), K1)

	def value(self, p, q, r):
		"""
		Calculate the value of the polynomial R[p][q] at the given r.
		"""
		if(p <= self.maxP and abs(q) <= p):
			return polyval(r, self.R[p][abs(q)])
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
			return [0]

def polyadd(_p1, _p2):
	p1 = _p1.copy()
	p2 = _p2.copy()
	if len(p1) >= len(p2):
		for i in range(len(p2)):
			p1[i] += p2[i]
		return p1
	else:
		for i in range(len(p1)):
			p2[i] += p1[i]
		return p2

def constmul(c, _p):
	p = _p.copy()
	for i in range(len(p)):
		p[i] *= c
	return p

def constdiv(_p, c):
	p = _p.copy()
	for i in range(len(p)):
		p[i] = p[i] // c
	return p

def polysub(_p1, _p2):
	p1 = _p1.copy()
	p2 = _p2.copy()
	p3 = constmul(-1, p2)
	return polyadd(p1, p3)

def polymul(_p1, _p2):
	p1 = _p1.copy()
	p2 = _p2.copy()
	m = len(p1)
	n = len(p2)
	prod = []
	for i in range(m + n - 1):
		prod.append(0)
   
	for i in range(m):
		for j in range(n):
			prod[i+j] += p1[i]*p2[j]
	while prod[-1] == 0:
		prod.pop(-1)
	return prod

def polyval(x, _p):
	p = _p.copy()
	n = len(p)
	result = p[-1]
	for i in range(n - 2, -1, -1):
		result = result * x + p[i]
	return result