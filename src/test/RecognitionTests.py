import argparse
import numpy as np
import os
from enum import Enum

from QZMI import *
from QZMRI import *
from legendre.QZMILegendre import *
from legendre.QZMRILegendre import *
from fourier.FourierMomentsMonochrome import *
from fourier.FourierMomentsColor import *
from fourier.FourierMomentsInvariantColor import *
from fourier.TransformationsFourier import *
from ImageManipulation import *
from Utility import *

class NoiseType(Enum):
	CLEAN = 1
	GAUSS = 2
	SALT = 3
	GAUSS_NO_ROUND = 4

class TestType(Enum):
	COIL_ROTATED = 1
	COIL_TRANSFORMED = 2
	CUPS_TRANSFORMED = 3

class QZMIType(Enum):
	NORMAL = 1
	LEGENDRE1 = 2
	LEGENDRE2 = 3
	FOURIER_INT = 4
	FOURIER_ORIGINAL = 5

class BGColor(Enum):
	BLACK = 1
	GREY = 2

def runRecognitionTest():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', '-f', required=True, type=str,
						help='The file where results are printed, without extension')
	parser.add_argument('--append', '-a', required=False, action='store_true',
						help='When specified, append to the file, otherwise overwrite it')
	args = parser.parse_args()
	file = args.file
	append = args.append

	if not append:
		open(file + ".txt", "w").close()
		open(file + ".err", "w").close()

	bgColor = BGColor.BLACK

	tests = [
		# (QZMIType.NORMAL, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.NORMAL, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.NORMAL, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.NORMAL, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.NORMAL, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.NORMAL, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.NORMAL, TestType.COIL_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.NORMAL, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.NORMAL, TestType.COIL_ROTATED, NoiseType.CLEAN),
		(QZMIType.NORMAL, TestType.COIL_ROTATED, NoiseType.GAUSS),
		(QZMIType.NORMAL, TestType.COIL_ROTATED, NoiseType.GAUSS_NO_ROUND),
		(QZMIType.NORMAL, TestType.COIL_ROTATED, NoiseType.SALT),

		# (QZMIType.LEGENDRE1, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE1, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE1, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.LEGENDRE1, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE1, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE1, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE1, TestType.COIL_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.LEGENDRE1, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE1, TestType.COIL_ROTATED, NoiseType.CLEAN),
		(QZMIType.LEGENDRE1, TestType.COIL_ROTATED, NoiseType.GAUSS),
		(QZMIType.LEGENDRE1, TestType.COIL_ROTATED, NoiseType.GAUSS_NO_ROUND),
		(QZMIType.LEGENDRE1, TestType.COIL_ROTATED, NoiseType.SALT),

		# (QZMIType.LEGENDRE2, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE2, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE2, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.LEGENDRE2, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE2, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.LEGENDRE2, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.LEGENDRE2, TestType.COIL_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.LEGENDRE2, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.LEGENDRE2, TestType.COIL_ROTATED, NoiseType.CLEAN),
		(QZMIType.LEGENDRE2, TestType.COIL_ROTATED, NoiseType.GAUSS),
		(QZMIType.LEGENDRE2, TestType.COIL_ROTATED, NoiseType.GAUSS_NO_ROUND),
		(QZMIType.LEGENDRE2, TestType.COIL_ROTATED, NoiseType.SALT),

		# (QZMIType.FOURIER_INT, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.FOURIER_INT, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.FOURIER_INT, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.FOURIER_INT, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.FOURIER_INT, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.FOURIER_INT, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.FOURIER_INT, TestType.COIL_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.FOURIER_INT, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.FOURIER_INT, TestType.COIL_ROTATED, NoiseType.CLEAN),
		(QZMIType.FOURIER_INT, TestType.COIL_ROTATED, NoiseType.GAUSS),
		(QZMIType.FOURIER_INT, TestType.COIL_ROTATED, NoiseType.GAUSS_NO_ROUND),
		(QZMIType.FOURIER_INT, TestType.COIL_ROTATED, NoiseType.SALT),

		# (QZMIType.FOURIER_ORIGINAL, TestType.CUPS_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.FOURIER_ORIGINAL, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.FOURIER_ORIGINAL, TestType.CUPS_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.FOURIER_ORIGINAL, TestType.CUPS_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.FOURIER_ORIGINAL, TestType.COIL_TRANSFORMED, NoiseType.CLEAN),
		# (QZMIType.FOURIER_ORIGINAL, TestType.COIL_TRANSFORMED, NoiseType.GAUSS),
		# (QZMIType.FOURIER_ORIGINAL, TestType.COIL_TRANSFORMED, NoiseType.GAUSS_NO_ROUND),
		# (QZMIType.FOURIER_ORIGINAL, TestType.COIL_TRANSFORMED, NoiseType.SALT),

		# (QZMIType.FOURIER_ORIGINAL, TestType.COIL_ROTATED, NoiseType.CLEAN),
		(QZMIType.FOURIER_ORIGINAL, TestType.COIL_ROTATED, NoiseType.GAUSS),
		(QZMIType.FOURIER_ORIGINAL, TestType.COIL_ROTATED, NoiseType.GAUSS_NO_ROUND),
		(QZMIType.FOURIER_ORIGINAL, TestType.COIL_ROTATED, NoiseType.SALT),
	]

	for (qzmiType, testType, noiseType) in tests:
		testRecognition(noiseType, testType, qzmiType, file, bgColor)

def getBasicRecognitionTestingData(testType, bgColor):
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
	
	if bgColor == BGColor.GREY:
		recognizePath = recognizePath[:-1] + "_grey/"
		originalPath = originalPath[:-1] + "_grey/"

	# recognizeFiles = os.listdir(recognizePath)
	recognizeFiles = os.listdir(recognizePath)
	originalFiles = os.listdir(originalPath)

	correctnessFun = isRecognitionCorrect

	return (recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun)

def getQZMIClass(testType, qzmiType):
	isFourier = qzmiType == QZMIType.FOURIER_INT or qzmiType == QZMIType.FOURIER_ORIGINAL
	# If QZMI is used:
	if testType == TestType.COIL_ROTATED:
		if qzmiType == QZMIType.NORMAL:
			qzmiClass = QZMRI
		elif qzmiType == QZMIType.LEGENDRE1:
			qzmiClass = QZMRILegendre1
		elif qzmiType == QZMIType.LEGENDRE2:
			qzmiClass = QZMRILegendre2
		elif qzmiType == QZMIType.FOURIER_INT:
			qzmiClass = FourierMomentsRotationInvariantInterpolation
		elif qzmiType == QZMIType.FOURIER_ORIGINAL:
			qzmiClass = FourierMomentsRotationInvariantOriginal
		else:
			printerr("ERROR: unsupported qzmiType")
			return
	elif testType == TestType.COIL_TRANSFORMED:
		if qzmiType == QZMIType.NORMAL:
			qzmiClass = QZMI
		elif qzmiType == QZMIType.LEGENDRE1:
			qzmiClass = QZMILegendre1
		elif qzmiType == QZMIType.LEGENDRE2:
			qzmiClass = QZMILegendre2
		elif qzmiType == QZMIType.FOURIER_INT:
			qzmiClass = FourierMomentsInvariantInterpolation
		elif qzmiType == QZMIType.FOURIER_ORIGINAL:
			qzmiClass = FourierMomentsInvariantOriginal
		else:
			printerr("ERROR: unsupported qzmiType")
			return
	elif testType == TestType.CUPS_TRANSFORMED:
		if qzmiType == QZMIType.NORMAL:
			qzmiClass = QZMI
		elif qzmiType == QZMIType.LEGENDRE1:
			qzmiClass = QZMILegendre1
		elif qzmiType == QZMIType.LEGENDRE2:
			qzmiClass = QZMILegendre2
		elif qzmiType == QZMIType.FOURIER_INT:
			qzmiClass = FourierMomentsInvariantInterpolation
		elif qzmiType == QZMIType.FOURIER_ORIGINAL:
			qzmiClass = FourierMomentsInvariantOriginal
		else:
			printerr("ERROR: unsupported qzmiType")
			return
	else:
		printerr("ERROR: unsupported testType")
		return
	return isFourier, qzmiClass

def testRecognition(noiseType, testType, qzmiType, file, bgColor):
	logAll(file, '\n\n')
	logAll(file, qzmiType, testType, noiseType)
	# Needs to use OldTransformation in ZernikeMomentsColor with centroidTranslation applied in advance
	np.random.seed(0)

	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData(testType, bgColor)

	isFourier, qzmiClass = getQZMIClass(testType, qzmiType)

	if noiseType == NoiseType.CLEAN:
		noiseFun = lambda img : img

		result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun, isFourier)
		printResultOfRecognition("Noise-free", result, file)

	elif noiseType == NoiseType.GAUSS or noiseType == NoiseType.GAUSS_NO_ROUND:
		if testType == TestType.CUPS_TRANSFORMED:
			stddevs = [1,2,3,5,7,9,40,50,60]
		elif testType == TestType.COIL_TRANSFORMED:
			stddevs = [1,2,3,5,7,9,40,50,60]
		elif testType == TestType.COIL_ROTATED:
			# stddevs = [40,50,60,70,80,90,100,110,120]
			# stddevs = [80,90,100,110,120]
			stddevs = [140,160,180,200]
		else:
			printerr("ERROR: unsupported testType")
			return

		if noiseType == NoiseType.GAUSS:
			for stddev in stddevs:
				noiseFun = lambda img : addGaussianNoise(img, mean=0, stddev=stddev)

				result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun, isFourier)
				printResultOfRecognition("Gaussian noise with std dev {0}".format(stddev), result, file)

		elif noiseType == NoiseType.GAUSS_NO_ROUND:
			for stddev in stddevs:
				noiseFun = lambda img : addGaussianNoiseNoRounding(img, mean=0, stddev=stddev)

				result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun, isFourier)
				printResultOfRecognition("Gaussian noise (no rounding) with std dev {0}".format(stddev), result, file)

	elif noiseType == NoiseType.SALT:
		if testType == TestType.CUPS_TRANSFORMED:
			densities = [0.2, 0.4, 0.6,1,2,3,5,10,15]
		elif testType == TestType.COIL_TRANSFORMED:
			densities = [0.2, 0.4, 0.6,1,2,3,5,10,15]
		elif testType == TestType.COIL_ROTATED:
			# densities = [5, 10, 15, 20, 25, 30]
			densities = [40, 50, 60, 75]
		else:
			printerr("ERROR: unsupported testType")
			return
		
		for density in densities:
			noiseFun = lambda img : addSaltAndPepperNoise(img, density=density)

			result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun, isFourier)
			printResultOfRecognition("Salt and pepper noise with density {0}%".format(density), result, file)
	
	else:
		printerr("ERROR: unsupported noiseType")

def recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, qzmiClass, correctnessFun, noiseFun=None, isFourier=False):
	originalVecs = {}
	recognizeVecs = {}

	for file in originalFiles:
		(img, _) = getImgFromFileAsNpArray(originalPath + file)
		if not isFourier:
			originalVecs[file] = populateInvariantVector(img, qzmiClass, noiseFun)
		else:
			img = np.array(img, dtype='double')
			originalVecs[file] = populateInvariantVectorFourier(img, noiseFun, qzmiClass)
	
	for file in recognizeFiles:
		(img, _) = getImgFromFileAsNpArray(recognizePath + file)
		if not isFourier:
			recognizeVecs[file] = populateInvariantVector(img, qzmiClass, noiseFun)
		else:
			img = np.array(img, dtype='double')
			recognizeVecs[file] = populateInvariantVectorFourier(img, noiseFun, qzmiClass)

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

