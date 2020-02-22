import numpy as np
import os

from QZMI import *
from ImageManipulation import *
from Utility import *

def getBasicRecognitionTestingData():
	# recognizePath = "../images/cups/transformed/"
	# recognizePath = "../images/coil/rotated/"
	recognizePath = "../images/coil/transformed/"
	# recognizeFiles = ["36x8y5r240s1_25.png", "262x8y5r30s0_5.png", "125x8y5r180s1_75.png"]
	# recognizeFiles = ["157x8y5r300s1_75.png", "157x8y5r180s1_25.png", "262x8y5r240s0_75.png", "259x8y5r300s2.png"]
	recognizeFiles = os.listdir(recognizePath)[::20]

	# originalPath = "../images/cups/extended/"
	originalPath = "../images/coil/extended/"
	originalFiles = os.listdir(originalPath)

	correctnessFun = isRecognitionCorrect

	return (recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun)


def testRecognition_Clean():
	# Needs to use CentroidTransformation in ZernikeMomentsColor

	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData()

	transformationFun = lambda img : img

	result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun, transformationFun)
	printResultOfRecognition("Noise-free", result)


def testRecognition_Gauss():
	# Needs to use CentroidTransformation in ZernikeMomentsColor
	
	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData()

	stddevs = [1, 2, 3, 5, 7, 9, 20, 40, 50, 60]
	for stddev in stddevs:
		transformationFun = lambda img : addGaussianNoise(img, mean=0, stddev=stddev)

		result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun, transformationFun)
		printResultOfRecognition("Gaussian noise with std dev {0}".format(stddev), result)


def testRecognition_SaltAndPepper():
	# Needs to use CentroidTransformation in ZernikeMomentsColor
	
	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData()

	densities = [0.2, 0.4, 0.6, 1, 2, 3, 5, 10, 15]
	for density in densities:
		transformationFun = lambda img : addSaltAndPepperNoise(img, density=density)

		result = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun, transformationFun)
		printResultOfRecognition("Salt and pepper noise with density {0}%".format(density), result)

def printResultOfRecognition(name, result):
	(correct, incorrect, pct) = result
	print("\n" + name)
	print(pct, "% recognized correctly.", flush=True)
	printerr("\n" + name)
	printerr(pct, "% recognized correctly.")
	if len(incorrect) > 0:
		printerr("Incorrect results: ")
	for (file, resultFile) in incorrect:
		printerr(file, "recognized as: ", resultFile)
	printerr("\n", flush=True)

def isRecognitionCorrect(fileToRecognize, recognizedFile):
	id = recognizedFile.split('.')[0]
	return fileToRecognize.startswith(id)

def recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun, transformationFun=None):
	originalVecs = {}
	recognizeVecs = {}

	for file in originalFiles:
		originalVecs[file] = populateInvariantVector(originalPath + file)
	
	for file in recognizeFiles:
		recognizeVecs[file] = populateInvariantVector(recognizePath + file, transformationFun)

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

def populateInvariantVector(imgPath, transformationFun=None):
	relevantMoments = [(1,1,1), (2,0,0), (2,2,2), (3,1,1), (3,3,3), (4,0,0), (4,2,2), (4,4,4)]
	maxDeg = 4
	result = []
	(img, N) = getImgFromFile(imgPath, transformationFun)
	# im = Image.fromarray(img)
	# im.save('../test.bmp', "BMP")
	# if transformationFun is not None:
	# 	img = transformationFun(img)
	# 	# im = Image.fromarray(img)
	# 	# im.save('../test.bmp', "BMP")
	qzmi = QZMI(img, N, maxDeg)
	for relevantMoment in relevantMoments:
		n,m,k = relevantMoment
		# r = 1 + max(n-m, k-m)//2 # Number of moments used
		# r = 2 + max(n-m, k-m)//2  # Test this + see notes.txt
		# r = 2 + (n - m)//2 + (k - m)//2
		# r = max(n, max(m,k))
		r = 2

		normalizedInvs = [np.sign(i)*(abs(i)**(1.0/r)) for i in  qzmi.QZMIs[n,m,k]]

		if n == k:
			# If n == k only the real part contains information
			normalizedInvs = normalizedInvs[:1]

		result.extend(normalizedInvs)
	return result