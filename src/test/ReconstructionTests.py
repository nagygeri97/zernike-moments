import argparse

from ZernikeMomentsColor import *
from ZernikeMomentsMonochrome import *
from Utility import *

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