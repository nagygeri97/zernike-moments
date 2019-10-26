#!/usr/bin/env python3

import argparse
import numpy as np
import quaternion
from PIL import Image

from RadialPolynomials import RadialPolynomials
from Transformations import *
from ZernikeMoments import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', '-f', required=True, type=str,
						help='The path to the image you want to process')
	parser.add_argument('--greyscale', '-g', action='store_true', required=False,
						help='Specify this flag to indicate that the selected image is greyscale.')
	args = parser.parse_args()

	image = Image.open(args.file)
	image.load()
	img = np.array(image)

	(N, M, _) = img.shape # img should be square
	z = ZernikeMomentsMonochrome(img, N, 5)

if __name__ == '__main__':
	main()