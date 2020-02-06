#!/usr/bin/env python3

import argparse
import numpy as np
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
	# testRecognition()
	# addGaussianNoiseAndPrintImage()
	# addSaltAndPepperNoiseAndPrintImage()

	stop = timeit.default_timer()
	# print('Time:', stop - start, "s")  

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

def testRecognition():
	recPath = "../images/cups/transformed/"
	imgsToRecognize = ["36x8y5r240s1_25.png", "262x8y5r30s0_5.png", "125x8y5r180s1_75.png"] 
	originalPath = "../images/cups/extended/"
	files = os.listdir(originalPath)
	originalVecs = {}
	vecsToRecognize = {}

	for file in files:
		originalVecs[file] = populateInvariantVector(originalPath+file)
	
	for file in imgsToRecognize:
		vecsToRecognize[file] = populateInvariantVector(recPath+file)

	for recFile, recVec in vecsToRecognize.items():
		minDist = -1
		minFile = ""
		for file, vec in originalVecs.items():
			dist = vectorDistance(vec, recVec)
			if minDist == -1 or dist < minDist:
				minDist = dist
				minFile = file
	
		print(recFile, "recognized as:", minFile)

def populateInvariantVector(imgPath):
	relevantMoments = [(1,1,1), (2,0,0), (2,2,2), (3,1,1), (3,3,3), (4,0,0), (4,2,2), (4,4,4)]
	maxDeg = 4
	result = []
	(img, N) = getImgFromFile(imgPath)
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

def getImgFromFile(fileName):
	image = Image.open(fileName)
	image.load()
	image = squareImage(image)
	img = np.array(image) # by this point img is assumed to be square
	(N, _, _) = img.shape

	return (img, N)

if __name__ == '__main__':
	main()