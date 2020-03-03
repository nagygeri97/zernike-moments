import numpy as np
import os

from QZMI import *
from legendre.QZMILegendre import *
from legendre.TransformationsLegendre import *
from Utility import *

def testInvariance():
	# Needs to use OldTransformation in ZernikeMomentsColor with centroidTranslation applied in advance

	path = "../images/cups/transformed/"
	prefix = "36"
	files = [file for file in os.listdir(path) if file.startswith(prefix)]
	maxDeg = 4
	values = {}
	for file in files:
		(img, N) = getImgFromFileAsNpArray(path+file)
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

def testLegendreInvariance():
	path = "../images/cups/transformed/"
	prefix = "161"
	files = [file for file in os.listdir(path) if file.startswith(prefix)]
	maxDeg = 4
	values = {}
	imgOrig, _ = getImgFromFileAsNpArray("../images/cups/extended/161.png")
	points = LegendrePoints1(imgOrig)
	for file in files:
		(img, N) = getImgFromFileAsNpArray(path+file)
		qzmi = QZMILegendre(img, N, maxDeg, points)
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