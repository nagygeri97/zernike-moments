import numpy as np
from numba import *
	
@jit(void(float64, int32, float64[:,:]), nopython=True)
def calculateRadialPolynomials(r, maxP, values):
	"""
	Create and store the values of the polynomials
	for a single given r,
	for p = 0..maxP, q =  0..p,
	where p - q is even

	Assumes r in [0..1]
	"""
	values[0,0] = 1
	values[1,1] = r
	for p in range(2,maxP + 1):
		h = p*(p - 1)*(p - 2)
		K2 = 2*h
		values[p,p] = (r**p)
		values[p,p-2] = p*values[p,p] - (p-1)*values[p-2,p-2]
		for q in range(p-4,-1,-2):			
			K1 = (p + q)*(p - q)*(p - 2)/2
			K3 = (-1)*q*q*(p - 1) - h
			K4 = (-1)*p*(p + q - 2)*(p - q - 2)/2

			r2 = r**2
			values[p,q] = ((K2*r2 + K3)*values[p-2,q] + K4*values[p-4,q]) / K1
	# return values
