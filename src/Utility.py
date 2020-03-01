import numpy as np
from PIL import Image
import sys

import ImageManipulation as IM
from Transformations import *

# ------ Logging ---------

def printerr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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