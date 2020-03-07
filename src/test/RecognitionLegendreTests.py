import numpy as np
import os
from enum import Enum

from legendre.QZMILegendre import *
from QZMRI import *
from ImageManipulation import *
from Utility import *
from test.RecognitionTests import *
from legendre.TransformationsLegendre import *



def getBasicRecognitionTestingDataLegendre(testType):
	if testType == TestType.COIL_ROTATED:
		recognizePath = "../images/coil/rotated/"
		originalPath = "../images/coil/extended/"
		qzmiClass = QZMRI # TODO: implement QZMRILegendre
	elif testType == TestType.COIL_TRANSFORMED:
		recognizePath = "../images/coil/transformed/"
		originalPath = "../images/coil/extended/"
		qzmiClass = QZMILegendre
	elif testType == TestType.CUPS_TRANSFORMED:
		recognizePath = "../images/cups/transformed/"
		originalPath = "../images/cups/extended/"
		qzmiClass = QZMILegendre
	else:
		printerr("ERROR: unsupported testType")
		return
	
	recognizeFiles = os.listdir(recognizePath)[::20]
	originalFiles = os.listdir(originalPath)

	correctnessFun = isRecognitionCorrect

	return (recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun)

def testRecognitionLegendre(noiseType, testType):
	# Needs to use OldTransformation in ZernikeMomentsColor with centroidTranslation applied in advance

	(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun) = getBasicRecognitionTestingDataLegendre(testType)

	if noiseType == NoiseType.CLEAN:
		noiseFun = lambda img : img

		result = recognizeAllLegendre(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun)
		printResultOfRecognition("Noise-free", result)

	elif noiseType == NoiseType.GAUSS:
		if testType == TestType.CUPS_TRANSFORMED:
			stddevs = [1,2,3]
		elif testType == TestType.COIL_TRANSFORMED:
			stddevs = [5,7,9]
		elif testType == TestType.COIL_ROTATED:
			stddevs = [40,50,60]
		else:
			printerr("ERROR: unsupported testType")
			return

		for stddev in stddevs:
			noiseFun = lambda img : addGaussianNoise(img, mean=0, stddev=stddev)

			result = recognizeAllLegendre(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun)
			printResultOfRecognition("Gaussian noise with std dev {0}".format(stddev), result)

	elif noiseType == NoiseType.SALT:
		if testType == TestType.CUPS_TRANSFORMED:
			densities = [0.2, 0.4, 0.6]
		elif testType == TestType.COIL_TRANSFORMED:
			densities = [1, 2, 3]
		elif testType == TestType.COIL_ROTATED:
			densities = [5, 10, 15]
		else:
			printerr("ERROR: unsupported testType")
			return
		
		for density in densities:
			noiseFun = lambda img : addSaltAndPepperNoise(img, density=density)

			result = recognizeAllLegendre(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun)
			printResultOfRecognition("Salt and pepper noise with density {0}%".format(density), result)
	
	else:
		printerr("ERROR: unsupported noiseType")

def recognizeAllLegendre(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun=None):
	originalVecs = {}
	recognizeVecs = {}

	img, _ = getImgFromFileAsNpArray(originalPath + originalFiles[0])
	points = LegendrePoints1(img)

	for file in originalFiles:
		originalVecs[file] = populateInvariantVectorLegendre(originalPath + file, qzmiClass, noiseFun, points)
	
	for file in recognizeFiles:
		recognizeVecs[file] = populateInvariantVectorLegendre(recognizePath + file, qzmiClass, noiseFun, points)

	correct = []
	incorrect = []

	for recFile, recVec in recognizeVecs.items():
		minDist = -1
		minFile = ""
		for file, vec in originalVecs.items():
			dist = vectorDistance(vec, recVec)
			if minDist == -1 or dist < minDist: # now using min
				minDist = dist
				minFile = file
		if correctnessFun(recFile, minFile):
			correct.append((recFile, minFile))
		else:
			incorrect.append((recFile, minFile))
	
	pct = float(len(correct)) / float(len(recognizeFiles)) * 100
	return (correct, incorrect, pct)

def populateInvariantVectorLegendre(imgPath, qzmiClass, noiseFun, points):
	relevantMoments = [(1,1,1), (2,0,0), (2,2,2), (3,1,1), (3,3,3), (4,0,0), (4,2,2), (4,4,4)]
	maxDeg = 4
	result = []
	(img, N) = getImgFromFileAsNpArray(imgPath)

	qzmi = qzmiClass(img, N, maxDeg, points, noiseFun)
	for relevantMoment in relevantMoments:
		n,m,k = relevantMoment
		r = 2

		normalizedInvs = [np.sign(i)*(abs(i)**(1.0/r)) for i in  qzmi.QZMIs[n,m,k]]

		if n == k:
			# If n == k only the real part contains information
			normalizedInvs = normalizedInvs[:1]

		result.extend(normalizedInvs)
	return result