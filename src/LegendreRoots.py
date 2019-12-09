import numpy as np
from numpy.polynomial.polynomial import *

def calculateLegendreRoots(n):
	"""
	Calculate the roots of the Legendre polynomial of degree n.
	Returns only the non-negative roots of Pn. 
	"""
	roots = []
	p = [1,0,-1]
	q = [0,-2,0]
	r = [n*(n+1),0,0]
	if n % 2 == 0:
		root = alg2(p,q,r,0,legendreValueAtZero(n))
	else:
		roots.append(0)

def alg2(p, q, r, xe, uxe):
	rder = polyder(r)
	pder = polyder(p)
	rderp = polymul(rder,p)
	pderr = polymul(pder,r)
	rq = polymul(r,q)
	rp = polymul(r,p)

	sqrtfn = lambda y : np.sqrt(polyval(y,r)/polyval(y,p))
	divfn = lambda y : (polyval(y, rderp) - polyval(y, pderr) + 2*polyval(y, rq)) / (2*polyval(y, rp))
	sinfn = lambda x : np.sin(2*x)/2
	f = lambda x,y: -(sqrtfn(y) + divfn(y)*sinfn(x))**(-1)
	
	x1 = RK2(0, -(np.pi/2), f, xe, 1000)
	
	ders = allDerivatives(30, xe, uxe, 0, p, q, r)

	taylor = taylorPolynomial(ders)
	dertaylor = polyder(taylor)
	x1 = newton(30, x1, taylor, dertaylor)
	return x1

def RK2(x0, L, f, y0, n):
	h = float(L)/n
	xi = x0
	yi = y0
	for i in range(n):
		k1 = h*f(xi,yi)
		k2 = h*f(xi + h, yi + k1)
		xi = xi + h
		yi = yi + 0.5*(k1 + k2)
	return yi

def newton(n, x0, taylor, dertaylor):
	xn = x0
	for i in range(n):
		xn = xn - polyval(xn, taylor) / polyval(xn, dertaylor)
	return xn

def taylorPolynomial(ders):
	#TODO
	pass

def legendreValueAtZero(n, all=False):
	vals = []
	if n % 2 == 0:
		P = 1.0
		nf = 2.0
		vals.append(P)
		for i in range(n//2):
			P = -((nf - 1)/nf)*P
			nf += 2
			vals.append(P)
		if all:
			return vals
		else:
			return P
	else:
		return 0.0

def legendreDerValueAtZero(n):
	if n % 2 == 0:
		return 0.0
	else:
		valsAtZero = legendreValueAtZero(n - 1, True)
		nf = 3.0
		Pder = 1.0
		for i in range(n//2):
			Pder = ((2*nf - 1)/nf) * valsAtZero[i + 1] - ((nf - 1)/nf)*Pder
			nf += 2
		return Pder

def legendreValue(n, x):
	nf = float(n)
	x = float(x)
	Pprev, Pcurr = 0, 1
	for i in range(n):
		Pprev, Pcurr = Pcurr,  ((2*nf - 1)/(nf))*x*Pcurr - ((nf - 1)/(nf))*Pprev
		nf -= 1
	return Pcurr

def allDerivatives(n, x, zeroth, first, p, q, r):
	"""
	Return the first n derivatives (n >= 4)
	"""
	ders = [zeroth, first]
	pd, qd, rd = polyder(p), polyder(q), polyder(r)
	pdd, qdd, rdd = polyder(pd), polyder(qd), polyder(rd)
	pv, qv, rv = polyval(x,p), polyval(x,q), polyval(x,r)
	pdv, qdv, rdv = polyval(x,pd), polyval(x,qd), polyval(x,rd)
	pddv, qddv, rddv = polyval(x,pdd), polyval(x,qdd), polyval(x,rdd)
	# k = 0
	second = -qv*first - rv*zeroth
	ders.append(second)
	# k = 1
	third = -(pdv + qv)*second - (qdv + rv)*first - rdv*zeroth
	ders.append(third)
	for k in range(2, n - 1):
		tmpk = float(k*(k-1))/2
		k2th = (-(k*pdv + qv)*ders[-1] - (tmpk*pddv + k*qdv + rv)*ders[-2]
			    -(tmpk*qddv + k*rdv)*ders[-3] - tmpk*rddv*ders[-4])
		ders.append(k2th)
	return ders