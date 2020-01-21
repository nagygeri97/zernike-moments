import numpy as np
from numba import *
from quaternion import *

from ZernikeMomentsColor import *

class QZMI:
	def __init__(self, img, N, maxP):
		self.img = img
		self.N = N
		self.maxP = maxP
		self.ZM = ZernikeMomentsColor(img, N, maxP)
		self.calculateQZMI()

	def calculateQZMI(self):
		self.calculateLs()
		self.QZMIs = np.zeros([self.maxP + 1, self.maxP + 1, self.maxP + 1, 4]) # Last index: re, i, j, k
		for n in range(0, maxP + 1):
			for k in range(0, n + 1):
				if (n - k) % 2 == 0:
					continue
				for m in range(0, k + 1):
					if(k - m) % 2 == 0:
						continue
					a = quaternion(*(self.Ls[n,m,i] for i in range(4)))
					b = quaternion(*(self.Ls[k,m,i] for i in range(4)))
					b = b.conj()
					qzmi = a * b
					self.QZMIs[n,m,k,0] = qzmi.w
					self.QZMIs[n,m,k,1] = qzmi.x
					self.QZMIs[n,m,k,2] = qzmi.y
					self.QZMIs[n,m,k,3] = qzmi.z

	def calculateLs(self):
		Gamma = np.sqrt(qAbs(self.ZM.Zre[0,0,0], self.ZM.Zi[0,0,0], self.ZM.Zj[0,0,0], self.ZM.Zk[0,0,0]))
		self.Ls = np.zeros([self.maxP + 1, self.maxP + 1, 4]) # Last index: re, i, j, k
		for n in range(0,maxP + 1):
			for m in range(0,n + 1):
				if (m - n) % 2 == 0:
					continue
				l = (n - m)//2
				for t in range(0,l + 1):
					for k in range(t, l + 1):
						G = Gamma**((-1)*(m + 2*k + 2))
						CD = getCD(l, k, m, t)
						tmp = G*CD
						self.Ls[n,m,0] += tmp * self.ZM.Zre[m + 2*t, m, 0]
						self.Ls[n,m,1] += tmp * self.ZM.Zi[m + 2*t, m, 0]
						self.Ls[n,m,2] += tmp * self.ZM.Zj[m + 2*t, m, 0]
						self.Ls[n,m,3] += tmp * self.ZM.Zk[m + 2*t, m, 0]

def getCD(l, k, m, t):
	"""
	Returns c*d as defined in (27) (29) (31)
	"""
	x = np.math.factorial(m + k + l)
	y = np.math.factorial(k - t)
	z = np.math.factorial(m + k + t + 1)
	w = 1 if (l - k % 2 == 0) else (-1)
	return w * (m + 2*l + 1) * float(x) / float(y * z)

def qAbs(re, i, j, k):
	return np.sqrt(re*re + i*i + j*j + k*k)