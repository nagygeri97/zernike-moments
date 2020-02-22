import numpy as np
import os

from QZMI import *
from Utility import *

def testInvariance():
	# Needs to use CentroidTransformation in ZernikeMomentsColor

	path = "../images/cups/transformed/"
	prefix = "36"
	files = [file for file in os.listdir(path) if file.startswith(prefix)]
	maxDeg = 4
	values = {}
	for file in files:
		(img, N) = getImgFromFile(path+file)
		qzmi = QZMI(img, N, maxDeg)
		for n in range(0, maxDeg + 1):
			for k in range(0, n + 1):
				if (n - k) % 2 != 0:
					continue
				for m in range(0, k + 1):
					if(k - m) % 2 != 0:
						continue
					key = "_".join((str(n),str(m),str(k)))
					if key not in values.keys():
						values[key] = []
					values[key].append(qAbs(*qzmi.QZMIs[n,m,k]))
	
	print("",*files, "Mean", "StDev", "StDev/Mean", sep=",")
	for key, value in values.items():
		mean = np.mean(value)
		stdev = np.std(value)
		print(key,*value, mean, stdev, stdev/mean, sep=",")