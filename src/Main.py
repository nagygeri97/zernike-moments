#!/usr/bin/env python3

import argparse
import numpy as np
import quaternion
from mpmath import *
from PIL import Image

from RadialPolynomials import RadialPolynomials
from Transformations import *
from ZernikeMoments import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', '-f', required=True, type=str,
						help='The path to the image you want to process')
	parser.add_argument('-M', required=False, type=int,
						help='Max number')
	parser.add_argument('-p', required=False, type=int,
						help='Precision')
	parser.add_argument('--greyscale', '-g', action='store_true', required=False,
						help='Specify this flag to indicate that the selected image is greyscale.')
	args = parser.parse_args()
	M = args.M if args.M is not None else 40
	mp.dps = args.p if args.p is not None else 10

	image = Image.open(args.file)
	image.load()
	img = np.array(image)

	(N, _, _) = img.shape # img should be square

	z = ZernikeMomentsMonochrome(img, N, M)

	z.reconstructImage("../test.bmp")

if __name__ == '__main__':
	main()