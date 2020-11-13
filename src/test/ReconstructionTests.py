import argparse
import csv

from ZernikeMomentsColor import *
from ZernikeMomentsMonochrome import *
from legendre.ZernikeMomentsColorLegendre import *
from legendre.TransformationsLegendre import *
from fourier.FourierMomentsMonochrome import *
from fourier.FourierMomentsColor import *
from fourier.TransformationsFourier import *
from Utility import *

def parseArgsForReconstructionTest():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', '-f', required=True, type=str,
						help='The path to the image you want to process')
	parser.add_argument('--output', '-o', required=False, type=str,
						help='The path/name of the output image')
	parser.add_argument('-M', required=True, type=int,
						help='Max number')
	parser.add_argument('--maxP', required=False, type=int,
						help='Max of Ms used over multiple testcases')
	parser.add_argument('--greyscale', '-g', action='store_true', required=False,
						help='Specify this flag to indicate that the selected image is greyscale.')
	args = parser.parse_args()
	return args

def testImageReconstructionTf1():
	# Needs to use OldTransformation in ZernikeMomentsColor/ZernikeMomentsMonochrome
	args = parseArgsForReconstructionTest()

	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)

	if args.greyscale:
		z = ZernikeMomentsMonochrome(getColorComponent(img), N, M)
		z.reconstructImage(output)
	else:
		z = ZernikeMomentsColorRight(img, N, M, OldTransformation)
		z.reconstructImage(output)

def testImageReconstructionTf2():
	# Needs to use OldTransformation in ZernikeMomentsColor/ZernikeMomentsMonochrome
	args = parseArgsForReconstructionTest()

	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)

	if args.greyscale:
		z = ZernikeMomentsMonochrome(getColorComponent(img), N, M)
		z.reconstructImage(output)
	else:
		z = ZernikeMomentsColorRight(img, N, M, OldTransformation2)
		z.reconstructImage(output)

def testImageReconstructionLegendre1():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	z = ZernikeMomentsColorRightLegendre(img, M, LegendreTransformation1)
	z.reconstructImage(output, N)

def testImageReconstructionLegendre2():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	z = ZernikeMomentsColorRightLegendre(img, M, LegendreTransformation2)
	z.reconstructImage(output, N)

def testImageReconstructionLegendreDiscOrth():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.png'
	M = args.M
	maxP = args.maxP if args.maxP is not None else M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	z = ZernikeMomentsColorRightLegendre(img, M, LegendreTransformationDiscOrth(maxP))
	z.reconstructImage(output, N)

def testImageReconstructionFourierMonochrome():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	f = FourierMomentsMonochrome(1, M, M, img=img)
	f.reconstructImage(output, N)

def testImageReconstructionFourierColor():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	f = FourierMomentsColor(img, M, M, FourierTransformationInterpolation, True)
	f.reconstructImage(output, N)

def testImageReconstructionFourierOriginalMonochrome():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	f = FourierMomentsMonochrome(1, M, M, trans=FourierTransformationOriginal(img, None), img=img)
	f.reconstructImage(output, N)

def testImageReconstructionFourierOriginalColor():
	args = parseArgsForReconstructionTest()
	
	output = args.output if args.output is not None else '../test.png'
	M = args.M

	(img, N) = getImgFromFileAsNpArray(args.file)
	img = np.array(img, dtype='double')

	f = FourierMomentsColor(img, M, M, FourierTransformationOriginal, True)
	f.reconstructImage(output, N)

def testImageReconstructionErrors():
	tests = [
		("lenna_color_64", [10,25,50,100,150,250,350]),
		("lenna_color_128", [10,25,50,100,150,250,350]),
		("lenna_color_256", [10,25,50,100,150,250,350]),
		("pepper_color_64", [10,25,50,100,150,250,350]),
		("pepper_color_128", [10,25,50,100,150,250,350]),
		("pepper_color_256", [10,25,50,100,150,250,350]),
	]
	path = "../images/lenna_pepper/"
	extension = ".bmp"
	tmpOut = "../tmp.png"

	parser = argparse.ArgumentParser()
	parser.add_argument('--output', '-o', required=True, type=str,
						help='The path to the result csv file')
	args = parser.parse_args()
	outFile = args.output

	with open(outFile, mode='w') as file:
		csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		csv_writer.writerow(['File', 'M', 'Original_tf1', 'Original_tf2', 'Legendre1', 'LegendreDiscOrth', 'Fourier'])
		for (file, Ms) in tests:
			print(file)
			filePath = path + file + extension
			(img, N) = getImgFromFileAsNpArray(filePath)
			maxP = max(Ms)
			for M in Ms:
				epss = []
				epss.append(ZernikeMomentsColorRight(img, N, M, OldTransformation).reconstructImage(tmpOut))
				epss.append(ZernikeMomentsColorRight(img, N, M, OldTransformation2).reconstructImage(tmpOut))
				img = np.array(img, dtype='double')
				epss.append(ZernikeMomentsColorRightLegendre(img, M, LegendreTransformation1).reconstructImage(tmpOut, N))
				epss.append(ZernikeMomentsColorRightLegendre(img, M, LegendreTransformationDiscOrth(maxP)).reconstructImage(tmpOut, N))
				epss.append(FourierMomentsColor(img, M, M, FourierTransformationInterpolation).reconstructImage(tmpOut, N))
				csv_writer.writerow((file, str(M), *epss))