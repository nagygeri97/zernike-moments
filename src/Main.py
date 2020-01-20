#!/usr/bin/env python3

import argparse
import numpy as np
import timeit
# import quaternion
from PIL import Image

from RadialPolynomials import *
from Transformations import *
from ZernikeMomentsMonochrome import *
from ZernikeMomentsColor import *
from ImageManipulation import *

def main():
	start = timeit.default_timer()

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

	image = Image.open(args.file)
	image.load()

	image = squareImage(image)

	img = np.array(image) # by this point img is assumed to be square
	(N, _, _) = img.shape

	# xc, yc = calculateCentroid(img)
	# print(xc,yc)

	# transformAndPrintImage(img, output)

	if args.greyscale:
		z = ZernikeMomentsMonochrome(getColorComponent(img), N, M)
		z.reconstructImage(output)
	else:
		z = ZernikeMomentsColorRight(img, N, M)
		z.reconstructImage(output)

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

if __name__ == '__main__':
	main()