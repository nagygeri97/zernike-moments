class RadialPolynomialValues:
	"""
	Class for calculating the Radial polynomials used in Zernike moment calculations.
	"""

	def __init__(self, maxP):
		self.maxP = maxP
	
	def calculateRadialPolynomials(self, r):
		"""
		Create and store the values of the polynomials
		for a single given r,
		for p = 0..maxP, q =  0..p,
		where p - q is even
		"""
		self.values = []
		for p in range(0,self.maxP + 1):
			self.values.append([])
			for q in range(0,p+1):
				self.values[p].append(0)

		for p in range(0,self.maxP + 1):
			for q in range(p,-1,-1):
				if ((p - q) % 2) == 1:
					self.values[p][q] = 0
					continue
				
				if p == q:
					self.values[p][q] = (r**p)
					continue
				
				if p - q == 2:
					self.values[p][q] = (q+2)*self.values[p][p] - (q+1)*self.values[q][q]
					continue

				K1 = (p + q)*(p - q)*(p - 2)/2
				K2 = 2*p*(p - 1)*(p - 2)
				K3 = (-1)*q*q*(p - 1) - p*(p - 1)*(p - 2)
				K4 = (-1)*p*(p + q - 2)*(p - q - 2)/2

				r2 = r**2
				self.values[p][q] = ((K2*r2 + K3)*self.values[p-2][q] + K4*self.values[p-4][q]) / K1

	def radPolyVal(self, p, q, r):
		if ((p - q) % 2) == 1:
			return 0
			
		if p == q:
			return (r**p)
		
		if p - q == 2:
			return (q+2)*self.radPolyVal(p, p, r) - (q+1)*self.radPolyVal(q, q, r)

		tmp1 = (q+2)*(r**(q+2)) - (q+1)*(r**q) # (q+2,q)
		tmp2 = (r**q) # (q,q)
		r2 = r**2
		for p2 in range(q + 4, p + 1, 2):
			K1 = (p2 + q)*(p2 - q)*(p2 - 2)/2
			K2 = 2*p2*(p2 - 1)*(p2 - 2)
			K3 = (-1)*q*q*(p2 - 1) - p2*(p2 - 1)*(p2 - 2)
			K4 = (-1)*p2*(p2 + q - 2)*(p2 - q - 2)/2

			tmp3 = ((K2*r2 + K3)*tmp1 + K4*tmp2) / K1
			tmp2 = tmp1
			tmp1 = tmp3

		return tmp1
