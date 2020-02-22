#!/usr/bin/env python3
import timeit

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
	# testRecognition_Clean()
	# testRecognition_Gauss()
	# testRecognition_SaltAndPepper()

	# NoiseGenerationTests
	addGaussianNoiseAndPrintImage()
	# addSaltAndPepperNoiseAndPrintImage()

	stop = timeit.default_timer()
	print('Time:', stop - start, "s")  

if __name__ == '__main__':
	main()