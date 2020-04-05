import numpy as np
from PIL import Image
import sys

from QZMI import *
from QZMRI import *
from legendre.QZMILegendre import *
from legendre.QZMRILegendre import *
import ImageManipulation as IM
from Transformations import *

# ------ Logging ---------

def printerr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def logAll(file, *args, **kwargs):
	outFileName = file + ".txt"
	errFileName = file + ".err"
	outFile = open(outFileName, 'a+')
	errFile = open(errFileName, 'a+')
	print(*args, file=outFile, **kwargs)
	print(*args, file=errFile, **kwargs)
	outFile.close()
	errFile.close()

def logError(file, *args, **kwargs):
	errFileName = file + ".err"
	errFile = open(errFileName, 'a+')
	print(*args, file=errFile, **kwargs)
	errFile.close()

# ------ Vector operations ---------

def normalizeVector(vec):
	length = vectorNorm(vec)
	result = [float(v) / length for v in vec]
	return result

def vectorNorm(vec):
	return np.linalg.norm(vec)

def vectorDistance(vec1, vec2):
	return np.linalg.norm(np.array(vec1) - np.array(vec2))

def dotDistance(vec1, vec2):
	return np.dot(vec1, vec2)

# ------ Image ---------

def getImgFromFileAsPILImg(fileName):
	image = Image.open(fileName)
	image.load()
	image = IM.squareImage(image)
	N, _ = image.size

	return (image, N)

def getImgFromFileAsNpArray(fileName):
	image = Image.open(fileName)
	image.load()
	image = IM.squareImage(image)
	img = np.array(image) # by this point img is assumed to be square
	(N, _, _) = img.shape

	return (img, N)

def getImgFromFileAsRawNpArray(fileName):
	image = Image.open(fileName)
	image.load()
	# NO SQUARING
	img = np.array(image)
	return img

def saveImgFromNpArray(img, fileName="../test.bmp"):
	image = Image.fromarray(img)
	image.save(fileName, "BMP")

def transformAndPrintImage(img, fileName):
	(N, _, _) = img.shape
	newImage = np.zeros((N, N, 3), dtype='uint8')
	for i in range(N):
		for j in range(N):
			newImage[i,j] = (255,255,255)
	trans = OldTransformation2(N, img)
	backTrans = ReverseTransformation(N, img)
	for x in range(N):
		for y in range(N):
			(r,theta) = trans.getPolarCoords(x,y)
			(nx, ny) = backTrans.getCartesianCoords(r,theta)
			if nx >= N or ny >= N:
				pass
			else:
				newImage[nx,ny] = img[x,y]
	im = Image.fromarray(newImage)
	im.save(fileName, "BMP")

def printImageFromLegendreTrans(trans, fileName="../test.bmp", n=1000):
	# n = trans.n
	newImage = np.zeros((n, n, 3), dtype='uint8')
	for k in range(trans.N):
		for j in range(4*trans.N + 1):
			xy = trans.rs[k] * (np.exp(trans.thetas[j]  * 1j))
			x = xy.real
			y = xy.imag

			x += 1
			y += 1

			x *= (n - 1) / 2
			y *= (n - 1) / 2

			x = int(round(x))
			y = int(round(y))

			for i in range(3):
				newImage[x,y,i] = trans.img[k,j,i]
	im = Image.fromarray(newImage)
	im.save(fileName, "BMP")

def calculateCentroid(img):
	m00 = 0
	m10 = 0
	m01 = 0
	(N, _, _) = img.shape
	for x in range(N):
		for y in range(N):
			s = sum([img[x,y,z] for z in range(3)])
			m10 += x*s
			m01 += y*s
			m00 += s
	m01 = int(round(float(m01) / m00))
	m10 = int(round(float(m10) / m00))
	return (m10, m01)

# ------ Recognition ---------

def isRecognitionCorrect(fileToRecognize, recognizedFile):
	id = recognizedFile.split('.')[0]
	return fileToRecognize.startswith(id)

def printResultOfRecognition(name, result, logFile):
	(correct, incorrect, pct) = result
	logAll(logFile, "\n" + name)
	logAll(logFile, pct, "% recognized correctly.", flush=True)
	if len(incorrect) > 0:
		logError(logFile, "Incorrect results: ")
	for (file, resultFile) in incorrect:
		logError(logFile, file, "recognized as: ", resultFile)
	logError(logFile, "\n", flush=True)

def populateInvariantVector(img, qzmiClass=QZMI, noiseFun=None):
	# img: an NxNx3 np.array
	relevantMoments = [(1,1,1), (2,0,0), (2,2,2), (3,1,1), (3,3,3), (4,0,0), (4,2,2), (4,4,4)]
	maxDeg = 4
	result = []

	(N,_,_) = img.shape

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