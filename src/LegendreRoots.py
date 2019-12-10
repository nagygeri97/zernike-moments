import numpy as np
from numpy.polynomial.polynomial import *

def calculateAllLegendreRoots(n):
	roots = []
	posRoots = calculateLegendreRoots(n)
	if n % 2 != 0:
		posRoots = posRoots[1:]
		roots.append(0.0)
	for root in posRoots:
		roots.append(-root)
		roots.append(root)
	roots.sort()
	return roots

def calculateLegendreRoots(n):
	"""
	Calculate the roots of the Legendre polynomial of degree n.
	Returns only the non-negative roots of Pn. 
	"""
	if n == 0:
		return []
	roots = []
	p = [1,0,-1]
	q = [0,-2,0]
	r = [n*(n+1),0,0]
	if n % 2 == 0:
		root = alg2(p,q,r,0,legendreValueAtZero(n),n)
		roots = alg1(p,q,r,root,n//2,legendreDerValue(n, root),n)
	else:
		roots = alg1(p,q,r,0.0,n//2 + 1, legendreDerValueAtZero(n),n)
	return roots

def getDiffEq(p, q, r):
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
	return f

def alg1(p, q, r, root, n, derroot, deg):
	roots = [root]
	ders = [derroot]
	f = getDiffEq(p, q, r)
	for i in range(1,n):
		xi = RK2((np.pi/2), -(np.pi), f, roots[-1], 30)
		
		ders = allDerivatives(30, roots[-1], 0, ders[-1], p, q, r, deg)
		(taylor, dertaylor) = taylorPolynomials(ders)
		xi = newton(30, xi, taylor, dertaylor, roots[-1])
		roots.append(xi)
		deri = legendreDerValue(deg, xi)
		ders.append(deri)
	return roots

def alg2(p, q, r, xe, uxe, deg):
	f = getDiffEq(p, q, r)

	x1 = RK2(0, -(np.pi/2), f, xe, 30)
	
	ders = allDerivatives(30, xe, uxe, 0, p, q, r, deg)

	(taylor, dertaylor) = taylorPolynomials(ders)
	x1 = newton(30, x1, taylor, dertaylor, xe)
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

def newton(n, x0, taylor, dertaylor, taylorcenter):
	xn = x0
	for i in range(n):
		xn = xn - polyval(xn - taylorcenter, taylor) / polyval(xn - taylorcenter, dertaylor)
	return xn

def taylorPolynomials(ders):
	fact = 1
	taylor = [ders[0]]
	dertaylor = []
	n = 1
	for der in ders[1:]:
		dertaylor.append(der / fact)
		fact *= n
		n += 1
		taylor.append(der / fact)
	return (taylor, dertaylor)

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

def legendreValue(n, x, all=False):
	nf = 0.0
	x = float(x)
	vals = [0,1]
	for i in range(n):
		tmp = ((2*nf + 1)/(nf + 1))*x*vals[-1] - ((nf)/(nf + 1))*vals[-2]
		vals.append(tmp)
		nf += 1
	if all:
		return vals
	else:
		return vals[-1]

def legendreDerValue(n, x):
	nf = 0.0
	x = float(x)
	values = legendreValue(n, x, True) # p(-1)(x), p(0)(x), p(1)(x),...,p(n)(x)
	vals = [0,0] 
	for i in range(n):
		tmp = ((2*nf + 1)/(nf + 1))*(x*vals[-1] + values[i + 1]) - ((nf)/(nf + 1))*vals[-2]
		vals.append(tmp)
		nf += 1
	return vals[-1]

def allDerivatives(n, x, zeroth, first, p, q, r, deg):
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
	second = (-qv*first - rv*zeroth) / pv
	ders.append(second)
	# k = 1
	third = (-(pdv + qv)*second - (qdv + rv)*first - rdv*zeroth) / pv
	ders.append(third)
	for k in range(2, n - 1):
		tmpk = float(k*(k-1))/2
		k2th = (-(k*pdv + qv)*ders[-1] - (tmpk*pddv + k*qdv + rv)*ders[-2]
			    -(tmpk*qddv + k*rdv)*ders[-3] - tmpk*rddv*ders[-4]) / pv
		if k + 2 > deg:
			ders.append(0.0)
		else:
			ders.append(k2th)
	return ders