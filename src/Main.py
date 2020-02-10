#!/usr/bin/env python3

import argparse
import numpy as np
import sys
import timeit
import os
from PIL import Image

from RadialPolynomials import *
from Transformations import *
from ZernikeMomentsMonochrome import *
from ZernikeMomentsColor import *
from QZMI import *
from ImageManipulation import *

def main():
	start = timeit.default_timer()

	# testImageReconstruction()
	# testInvariance()
	testRecognition_Clean()
	testRecognition_Gauss()
	testRecognition_SaltAndPepper()
	# addGaussianNoiseAndPrintImage()
	# addSaltAndPepperNoiseAndPrintImage()

	stop = timeit.default_timer()
	print('Time:', stop - start, "s")  

def transformAndPrintImage(img, fileName):
	(N, _, _) = img.shape
	newImage = np.zeros((N, N, 3), dtype='uint8')
	for i in range(N):
		for j in range(N):
			newImage[i,j] = (255,255,255)
	trans = OldTransformation(N, img)
	backTrans = ReverseTransformation(N, img)
	for x in range(N):
		for y in range(N):
			# print(x, y)
			(r,theta) = trans.getPolarCoords(x,y)
			(nx, ny) = backTrans.getCartesianCoords(r,theta)
			if nx >= N or ny >= N:
				pass
			else:
				newImage[nx,ny] = img[x,y]
	im = Image.fromarray(newImage)
	im.save(fileName, "BMP")

def addGaussianNoiseAndPrintImage():
	(img,_) = getImgFromFile('../images/cups/extended/36.png')
	newImg = addGaussianNoise(img,mean=0,stddev=3)
	im = Image.fromarray(newImg)
	im.save('../test.bmp', "BMP")
def addSaltAndPepperNoiseAndPrintImage():
	(img,_) = getImgFromFile('../images/cups/extended/36.png')
	newImg = addSaltAndPepperNoise(img,density=5)
	im = Image.fromarray(newImg)
	im.save('../test.bmp', "BMP")

def testImageReconstruction():
	# Needs to use OldTransformation in ZernikeMomentsColor/ZernikeMomentsMonochrome
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', '-f', required=True, type=str,
						help='The path to the image you want to process')
	parser.add_argument('--output', '-o', required=False, type=str,
						help='The path/name of the output image')
	parser.add_argument('-M', required=True, type=int,
						help='Max number')
	parser.add_argument('--greyscale', '-g', action='store_true', required=False,
						help='Specify this flag to indicate that the selected image is greyscale.')
	args = parser.parse_args()
	output = args.output if args.output is not None else '../test.bmp'
	M = args.M

	(img, N) = getImgFromFile(args.file)

	if args.greyscale:
		z = ZernikeMomentsMonochrome(getColorComponent(img), N, M)
		z.reconstructImage(output)
	else:
		z = ZernikeMomentsColorRight(img, N, M)
		z.reconstructImage(output)

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
		for n in range(1, maxDeg + 1):
			for k in range(1, n + 1):
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

def getBasicRecognitionTestingData():
	# recognizePath = "../images/cups/transformed/"
	recognizePath = "../images/coil/rotated/"
	# recognizePath = "../images/coil/transformed/"
	# recognizeFiles = ["36x8y5r240s1_25.png", "262x8y5r30s0_5.png", "125x8y5r180s1_75.png"]
	recognizeFiles = os.listdir(recognizePath)[:10:]

	# originalPath = "../images/cups/extended/"
	originalPath = "../images/coil/extended/"
	originalFiles = os.listdir(originalPath)

	correctnessFun = isRecognitionCorrect

	return (recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun)

def testRecognition_Clean():
	# Needs to use CentroidTransformation in ZernikeMomentsColor

	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData()

	transformationFun = lambda img : img

	(correct, incorrect, pct) = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun, transformationFun)
	print("\nNoise-free")
	print(pct, "% recognized correctly.")
	printerr("\nNoise-free")
	printerr(pct, "% recognized correctly.")
	if len(incorrect) > 0:
		printerr("Incorrect results: ")
	for (file, result) in incorrect:
		printerr(file, "recognized as: ", result)
	printerr("\n")

def testRecognition_Gauss():
	# Needs to use CentroidTransformation in ZernikeMomentsColor
	
	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData()

	stddevs = [1,2,3,5,7,9,20,40,50,60]
	for stddev in stddevs:
		transformationFun = lambda img : addGaussianNoise(img, mean=0, stddev=stddev)

		(correct, incorrect, pct) = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun, transformationFun)
		print("\nGaussian noise with std dev", stddev)
		print(pct, "% recognized correctly.")
		printerr("\nGaussian noise with std dev", stddev)
		printerr(pct, "% recognized correctly.")
		if len(incorrect) > 0:
			printerr("Incorrect results: ")
		for (file, result) in incorrect:
			printerr(file, "recognized as: ", result)
		printerr("\n")

def testRecognition_SaltAndPepper():
	# Needs to use CentroidTransformation in ZernikeMomentsColor
	
	(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun) = getBasicRecognitionTestingData()

	densities = [0.2, 0.4, 0.6, 1, 2, 3, 5, 10, 15]
	for density in densities:
		transformationFun = lambda img : addSaltAndPepperNoise(img, density=density)

		(correct, incorrect, pct) = recognizeAll(recognizePath, recognizeFiles, originalPath, originalFiles, correctnessFun, transformationFun)
		print("\nSalt and pepper noise with density", density, "%")
		print(pct, "% recognized correctly.")
		printerr("\nSalt and pepper noise with density", density, "%")
		printerr(pct, "% recognized correctly.")
		if len(incorrect) > 0:
			printerr("Incorrect results: ")
		for (file, result) in incorrect:
			printerr(file, "recognized as: ", result)
		printerr("\n")

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
			# print(recFile,file,dist)
			if minDist == -1 or dist < minDist:
				minDist = dist
				minFile = file
		if correctnessFun(recFile, minFile):
			correct.append((recFile, minFile))
		else:
			incorrect.append((recFile, minFile))
	
	pct = float(len(correct)) / float(len(recognizeFiles)) * 100
	return (correct, incorrect, pct)

def isRecognitionCorrect(fileToRecognize, recognizedFile):
	id = recognizedFile.split('.')[0]
	return fileToRecognize.startswith(id)

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
		r = 1 + max(n-m, k-m)//2 # Number of moments used
		normalizedInvs = [np.sign(i)*(abs(i)**(1.0/r)) for i in  qzmi.QZMIs[n,m,k]]
		if n == m and m == k:
			# If n == m == k only the real part contains information
			normalizedInvs = normalizedInvs[:1] 
		result.extend(normalizedInvs)
	return result

def vectorDistance(vec1, vec2):
	diffsq = [abs(v1-v2)**2 for v1,v2 in zip(vec1, vec2)]
	dist = np.sqrt(sum(diffsq))
	return dist

def getImgFromFile(fileName, transformationFun=None):
	image = Image.open(fileName)
	image.load()
	if transformationFun is not None:
		imarray = np.array(image)
		imarray = transformationFun(imarray)
		image = Image.fromarray(imarray)
	image = squareImage(image)
	img = np.array(image) # by this point img is assumed to be square
	(N, _, _) = img.shape

	return (img, N)

def printerr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
	main()