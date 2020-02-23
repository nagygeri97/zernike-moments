#!/usr/bin/env python3
import timeit

from Utility import *

from test.InvarianceTests import *
from test.NoiseGenerationTests import *
from test.RecognitionTests import *
from test.ReconstructionTests import *

def main():
	start = timeit.default_timer()

	# ReconstructionTests
	# testImageReconstruction()
	
	# InvarianceTests
	# testInvariance()

	# RecognitionTests
	testType = TestType.COIL_ROTATED
	testRecognition(NoiseType.CLEAN, testType)
	testRecognition(NoiseType.GAUSS, testType)
	testRecognition(NoiseType.SALT, testType)

	# NoiseGenerationTests
	# addGaussianNoiseAndPrintImage()
	# addSaltAndPepperNoiseAndPrintImage()

	# img, N = getImgFromFileAsNpArray("../images/cups/transformed/36x8y5r30s1_0.png")
	# centroidTranslation(img)

	stop = timeit.default_timer()
	print('Time:', stop - start, "s")  

if __name__ == '__main__':
	main()