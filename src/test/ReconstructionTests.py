import argparse

from ZernikeMomentsColor import *
from ZernikeMomentsMonochrome import *
from legendre.ZernikeMomentsColorLegendre import *
from legendre.TransformationsLegendre import *
from Utility import *

def parseArgsForReconstructionTest():
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
	return args

def testImageReconstruction():
	# Needs to use OldTransformation in ZernikeMomentsColor/ZernikeMomentsMonochrome
	args = parseArgsForReconstructionTest()

	output = args.output if args.output is not None else '../test.bmp'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)

	if args.greyscale:
		z = ZernikeMomentsMonochrome(getColorComponent(img), N, M)
		z.reconstructImage(output)
	else:
		z = ZernikeMomentsColorRight(img, N, M)
		z.reconstructImage(output)

def testImageReconstructionLegendre1():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.bmp'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	z = ZernikeMomentsColorRightLegendre(img, M, LegendreTransformation1)
	z.reconstructImage(output, N)

def testImageReconstructionLegendre2():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.bmp'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	z = ZernikeMomentsColorRightLegendre(img, M, LegendreTransformation2)
	z.reconstructImage(output, N)
