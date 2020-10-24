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

@jit(void(float64, int32, float64[:]), nopython=True)
def calculateFourierKernel(r, maxP, values):
	"""
	Create and store the values of the Fourier kernel functions
	for a single given r,
	for p = 0..maxP
	
	Assumes r in [0..1]
	"""
	# TODO: What if r == 0?
	if r < 1e-8:
		r = 1e-8

	values[0] = 1.0 / np.sqrt(r)
	sq2r = np.sqrt(2.0 / r)
	# Even p
	for p in range(2, maxP + 1, 2):
		values[p] = sq2r * np.cos(np.pi * p * r)
	
	# Odd p
	for p in range(1, maxP + 1, 2):
		values[p] = sq2r * np.sin(np.pi * (p + 1) * r)

@jit(void(float64, int32, float64[:]), nopython=True)
def calculateFourierKernelNegativeP(r, maxP, values):
	"""
	Create and store the values of the Fourier kernel functions
	for a single given r,
	for p = -maxP..0
	
	Assumes r in [0..1]
	"""
	# TODO: What if r == 0?
	if r < 1e-8:
		r = 1e-8

	values[0] = 1.0 / np.sqrt(r)
	sq2r = np.sqrt(2.0 / r)
	# Even p
	for p in range(2, maxP + 1, 2):
		values[p] = sq2r * np.cos(np.pi * (-p) * r)
	
	# Odd p
	for p in range(1, maxP + 1, 2):
		values[p] = sq2r * np.sin(np.pi * (-p + 1) * r)