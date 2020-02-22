import numpy as np
from PIL import Image
import sys

from ImageManipulation import *

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

# ------ Image IO ---------

def getImgFromFile(fileName, transformationFun=None):
	image = Image.open(fileName)
	image.load()
	image = squareImage(image)
	img = np.array(image) # by this point img is assumed to be square

	if transformationFun is not None:
		img = transformationFun(img)

	(N, _, _) = img.shape

	return (img, N)

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