import numpy as np
from numba import *
from quaternion import *

from ZernikeMomentsColor import *
from ImageManipulation import *
from Transformations import *
from Utility import *

class QZMI:
	"""
	Quaternion Zernike Moment Invariants
	RST invariant
	"""
	def __init__(self, img, N, maxP, noiseFun = None, transformation = None, centroidTranslate = True):
		self.img = img

		# self.img = centroidTranslationFloat(self.img)

		# Add noise before/after centroidTranslation
		if noiseFun is not None:
			self.img = noiseFun(self.img)

		if centroidTranslate:
			self.img = centroidTranslationFloat(self.img)

		# saveImgFromNpArray(img, "../original.png")
		# saveImgFromNpArray(self.img, "../test.png")

		self.N = N
		self.maxP = maxP
		self.ZM = ZernikeMomentsColorRight(self.img, self.N, self.maxP, transformation)
		self.calculateQZMI()

	def calculateQZMI(self):
		self.calculateLs()
		self.QZMIs = np.zeros([self.maxP + 1, self.maxP + 1, self.maxP + 1, 4]) # Last index: re, i, j, k
		for n in range(0, self.maxP + 1):
			for k in range(0, n + 1):
				if (n - k) % 2 != 0:
					continue
				for m in range(0, k + 1):
					if (k - m) % 2 != 0:
						continue
					a = quaternion(*self.Ls[n,m])
					b = quaternion(*self.Ls[k,m])
					b = b.conj()
					qzmi = a * b
					self.QZMIs[n,m,k,0] = qzmi.w
					self.QZMIs[n,m,k,1] = qzmi.x
					self.QZMIs[n,m,k,2] = qzmi.y
					self.QZMIs[n,m,k,3] = qzmi.z

	def calculateLs(self):
		Gamma = np.sqrt(qAbs(self.ZM.Zre[0,0], self.ZM.Zi[0,0], self.ZM.Zj[0,0], self.ZM.Zk[0,0]))
		self.Ls = np.zeros([self.maxP + 1, self.maxP + 1, 4]) # Last index: re, i, j, k
		for n in range(0, self.maxP + 1):
			for m in range(0,n + 1):
				if (m - n) % 2 != 0:
					continue
				l = (n - m)//2
				for t in range(0,l + 1):
					for k in range(t, l + 1):
						if Gamma != 0:
							G = Gamma**((-1)*(m + 2*k + 2))
						else:
							G = 0
						CD = getCD(l, k, m, t)
						tmp = G*CD
						self.Ls[n,m,0] += tmp * self.ZM.Zre[m + 2*t, m]
						self.Ls[n,m,1] += tmp * self.ZM.Zi[m + 2*t, m]
						self.Ls[n,m,2] += tmp * self.ZM.Zj[m + 2*t, m]
						self.Ls[n,m,3] += tmp * self.ZM.Zk[m + 2*t, m]

def getCD(l, k, m, t):
	"""
	Returns c*d as defined in (27) (29) (31)
	"""
	x = np.math.factorial(m + k + l)
	u = np.math.factorial(l - k)
	y = np.math.factorial(k - t)
	z = np.math.factorial(m + k + t + 1)
	w = 1 if ((l - k) % 2 == 0) else (-1)
	return w * (m + 2*l + 1) * float(x) / float(u * y * z)

def qAbs(re, i, j, k):
	return np.sqrt(re*re + i*i + j*j + k*k)

class QZMI_NoCentroid(QZMI):
	def __init__(self, img, N, maxP, noiseFun = None):
		super().__init__(img, N, maxP, noiseFun, OldTransformation, False)

class QZMI2_NoCentroid(QZMI):
	def __init__(self, img, N, maxP, noiseFun = None):
		super().__init__(img, N, maxP, noiseFun, OldTransformation2, False)

class QZMI2(QZMI):
	def __init__(self, img, N, maxP, noiseFun = None):
		super().__init__(img, N, maxP, noiseFun, OldTransformation2)