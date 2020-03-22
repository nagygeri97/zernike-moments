import numpy as np
from numba import *
from quaternion import *

from legendre.ZernikeMomentsColorLegendre import *
from ImageManipulation import *

class QZMRILegendre:
	"""
	Quaternion Zernike Moment ROTATION Invariants
	"""
	def __init__(self, img, N, maxP, transformationClass, noiseFun = None):
		self.img = img

		if noiseFun is not None:
			self.img = noiseFun(self.img)

		# self.img = centroidTranslationFloat(self.img)
		self.img = imageToFloat(self.img)

		self.N = N
		self.maxP = maxP
		self.ZM = ZernikeMomentsColorRightLegendre(self.img, self.maxP, transformationClass)
		self.calculateQZMRI()

	def calculateQZMRI(self):
		# Named QZMI for compatibility
		self.QZMIs = np.zeros([self.maxP + 1, self.maxP + 1, self.maxP + 1, 4]) # Last index: re, i, j, k
		for n in range(0, self.maxP + 1):
			for k in range(0, n + 1):
				if (n - k) % 2 != 0:
					continue
				for m in range(0, k + 1):
					if (k - m) % 2 != 0:
						continue
					a = quaternion(self.ZM.Zre[n, m], self.ZM.Zi[n, m], self.ZM.Zj[n, m], self.ZM.Zk[n, m])
					b = quaternion(self.ZM.Zre[k, m], self.ZM.Zi[k, m], self.ZM.Zj[k, m], self.ZM.Zk[k, m])
					b = b.conj()
					qzmri = (-1) * a * b
					self.QZMIs[n,m,k,0] = qzmri.w
					self.QZMIs[n,m,k,1] = qzmri.x
					self.QZMIs[n,m,k,2] = qzmri.y
					self.QZMIs[n,m,k,3] = qzmri.z

class QZMRILegendre1(QZMRILegendre):
	def __init__(self, img, N, maxP, noiseFun = None):
		super().__init__(img, N, maxP, LegendreTransformation1, noiseFun)

class QZMRILegendre2(QZMRILegendre):
	def __init__(self, img, N, maxP, noiseFun = None):
		super().__init__(img, N, maxP, LegendreTransformation2, noiseFun)