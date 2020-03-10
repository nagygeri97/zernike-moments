import numpy as np
import os
from enum import Enum

from QZMI import *
from QZMRI import *
from legendre.QZMILegendre import *
from legendre.QZMRILegendre import *
from ImageManipulation import *
from Utility import *

class NoiseType(Enum):
	CLEAN = 1
	GAUSS = 2
	SALT = 3

class TestType(Enum):
	COIL_ROTATED = 1
	COIL_TRANSFORMED = 2
	CUPS_TRANSFORMED = 3

class QZMIType(Enum):
	NORMAL = 1
	LEGENDRE1 = 2
	LEGENDRE2 = 3

def runRecognitionTest():

	tests = [
		(QZMIType.NORMAL, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		(QZMIType.NORMAL, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		(QZMIType.NORMAL, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.NORMAL, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.NORMAL, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.NORMAL, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.NORMAL, TestType.COIL_ROTATED, NoiseType.CLEAN),
		# (QZMIType.NORMAL, TestType.COIL_ROTATED, NoiseType.GAUSS),
		# (QZMIType.NORMAL, TestType.COIL_ROTATED, NoiseType.SALT),

		# (QZMIType.LEGENDRE1, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE1, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE1, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE1, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE1, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE1, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE1, TestType.COIL_ROTATED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE1, TestType.COIL_ROTATED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE1, TestType.COIL_ROTATED, NoiseType.SALT),

		# (QZMIType.LEGENDRE2, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE2, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE2, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE2, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE2, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE2, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE2, TestType.COIL_ROTATED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE2, TestType.COIL_ROTATED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE2, TestType.COIL_ROTATED, NoiseType.SALT),
	]

	for (qzmiType, testType, noiseType) in tests:
		testRecognition(noiseType, testType, qzmiType)

def getBasicRecognitionTestingData(testType):
	if testType == TestType.COIL_ROTATED:
		recognizePath = "../images/coil/rotated/"
		originalPath = "../images/coil/extended/"
	elif testType == TestType.COIL_TRANSFORMED:
		recognizePath = "../images/coil/transformed/"
		originalPath = "../images/coil/extended/"
	elif testType == TestType.CUPS_TRANSFORMED:
		recognizePath = "../images/cups/transformed/"
		originalPath = "../images/cups/extended/"
	else:
		printerr("ERROR: unsupported testType")
		return
	
	recognizeFiles = os.listdir(recognizePath)[::20]
	originalFiles = os.listdir(originalPath)

	correctnessFun = isRecognitionCorrect

	return (recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun)

def getQZMIClass(testType, qzmiType):
	if testType == TestType.COIL_ROTATED:
		if qzmiType == QZMIType.NORMAL:
			qzmiClass = QZMRI
		elif qzmiType == QZMIType.LEGENDRE1:
			qzmiClass = QZMRILegendre
		elif qzmiType == QZMIType.LEGENDRE2:
			qzmiClass = QZMRILegendre
		else:
			printerr("ERROR: unsupported qzmiType")
			return
	elif testType == TestType.COIL_TRANSFORMED:
		if qzmiType == QZMIType.NORMAL:
			qzmiClass = QZMI
		elif qzmiType == QZMIType.LEGENDRE1:
			qzmiClass = QZMILegendre
		elif qzmiType == QZMIType.LEGENDRE2:
			qzmiClass = QZMILegendre
		else:
			printerr("ERROR: unsupported qzmiType")
			return
	elif testType == TestType.CUPS_TRANSFORMED:
		if qzmiType == QZMIType.NORMAL:
			qzmiClass = QZMI
		elif qzmiType == QZMIType.LEGENDRE1:
			qzmiClass = QZMILegendre
		elif qzmiType == QZMIType.LEGENDRE2:
			qzmiClass = QZMILegendre
		else:
			printerr("ERROR: unsupported qzmiType")
			return
	else:
		printerr("ERROR: unsupported testType")
		return
	return qzmiClass

def testRecognition(noiseType, testType, qzmiType):
	print(qzmiType, testType, noiseType)
	printerr(qzmiType, testType, noiseType)
	# Needs to use OldTransformation in ZernikeMomentsColor with centroidTranslation applied in advance
	np.random.seed(0)

	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData(testType)
	qzmiClass = getQZMIClass(testType, qzmiType)

	if noiseType == NoiseType.CLEAN:
		noiseFun = lambda img : img

		result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun)
		printResultOfRecognition("Noise-free", result)

	elif noiseType == NoiseType.GAUSS:
		if testType == TestType.CUPS_TRANSFORMED:
			stddevs = [1,2,3,5,7,9,40,50,60]
		elif testType == TestType.COIL_TRANSFORMED:
			stddevs = [5,7,9]
		elif testType == TestType.COIL_ROTATED:
			stddevs = [40,50,60]
		else:
			printerr("ERROR: unsupported testType")
			return

		for stddev in stddevs:
			noiseFun = lambda img : addGaussianNoise(img, mean=0, stddev=stddev)

			result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun)
			printResultOfRecognition("Gaussian noise with std dev {0}".format(stddev), result)

	elif noiseType == NoiseType.SALT:
		if testType == TestType.CUPS_TRANSFORMED:
			densities = [0.2, 0.4, 0.6,1,2,3,5,10,15]
		elif testType == TestType.COIL_TRANSFORMED:
			densities = [1, 2, 3]
		elif testType == TestType.COIL_ROTATED:
			densities = [5, 10, 15]
		else:
			printerr("ERROR: unsupported testType")
			return
		
		for density in densities:
			noiseFun = lambda img : addSaltAndPepperNoise(img, density=density)

			result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun)
			printResultOfRecognition("Salt and pepper noise with density {0}%".format(density), result)
	
	else:
		printerr("ERROR: unsupported noiseType")

def recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun=None):
	originalVecs = {}
	recognizeVecs = {}

	for file in originalFiles:
		originalVecs[file] = populateInvariantVector(originalPath + file, qzmiClass, noiseFun)
	
	for file in recognizeFiles:
		recognizeVecs[file] = populateInvariantVector(recognizePath + file, qzmiClass, noiseFun)

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

def populateInvariantVector(imgPath, qzmiClass=QZMI, noiseFun=None):
	relevantMoments = [(1,1,1), (2,0,0), (2,2,2), (3,1,1), (3,3,3), (4,0,0), (4,2,2), (4,4,4)]
	maxDeg = 4
	result = []
	(img, N) = getImgFromFileAsNpArray(imgPath)

	qzmi = qzmiClass(img, N, maxDeg, noiseFun)
	for relevantMoment in relevantMoments:
		n,m,k = relevantMoment
		r = 2

		normalizedInvs = [np.sign(i)*(abs(i)**(1.0/r)) for i in  qzmi.QZMIs[n,m,k]]

		if n == k:
			# If n == k only the real part contains information
			normalizedInvs = normalizedInvs[:1]

		result.extend(normalizedInvs)
	return result