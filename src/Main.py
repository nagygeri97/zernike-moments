#!/usr/bin/env python3
import timeit

from Utility import *

from test.InvarianceTests import *
from test.LegendreTransformationTests import *
from test.NoiseGenerationTests import *
from test.RecognitionTests import *
from test.RecognitionLegendreTests import *
from test.ReconstructionTests import *

def main():
	start = timeit.default_timer()

	# ReconstructionTests
	# testImageReconstruction()
	
	# InvarianceTests
	# testInvariance()
	# testLegendreInvariance()

	# # RecognitionTests
	# print("\n\nOLD\n\n")
	# printerr("\n\nOLD\n\n")

	# print("\nCUPS_TRANSFORMED")
	# printerr("\nCUPS_TRANSFORMED")
	# testType = TestType.CUPS_TRANSFORMED
	# testRecognition(NoiseType.CLEAN, testType)
	# testRecognition(NoiseType.GAUSS, testType)
	# testRecognition(NoiseType.SALT, testType)

	# print("\nCOIL_TRANSFORMED")
	# printerr("\nCOIL_TRANSFORMED")
	# testType = TestType.COIL_TRANSFORMED
	# testRecognition(NoiseType.CLEAN, testType)
	# testRecognition(NoiseType.GAUSS, testType)
	# testRecognition(NoiseType.SALT, testType)

	# print("\nCOIL_ROTATED")
	# printerr("\nCOIL_ROTATED")
	# testType = TestType.COIL_ROTATED
	# testRecognition(NoiseType.CLEAN, testType)
	# testRecognition(NoiseType.GAUSS, testType)
	# testRecognition(NoiseType.SALT, testType)


	# print("\n\nLEGENDRE\n\n")
	# printerr("\n\nLEGENDRE\n\n")

	print("\nCUPS_TRANSFORMED")
	printerr("\nCUPS_TRANSFORMED")
	testType = TestType.CUPS_TRANSFORMED
	testRecognitionLegendre(NoiseType.CLEAN, testType)
	testRecognitionLegendre(NoiseType.GAUSS, testType)
	testRecognitionLegendre(NoiseType.SALT, testType)

	print("\nCOIL_TRANSFORMED")
	printerr("\nCOIL_TRANSFORMED")
	testType = TestType.COIL_TRANSFORMED
	testRecognitionLegendre(NoiseType.CLEAN, testType)
	testRecognitionLegendre(NoiseType.GAUSS, testType)
	testRecognitionLegendre(NoiseType.SALT, testType)

	# print("\nCOIL_ROTATED")
	# printerr("\nCOIL_ROTATED")
	# testType = TestType.COIL_ROTATED
	# testRecognitionLegendre(NoiseType.CLEAN, testType)
	# testRecognitionLegendre(NoiseType.GAUSS, testType)
	# testRecognitionLegendre(NoiseType.SALT, testType)

	# NoiseGenerationTests
	# addGaussianNoiseAndPrintImage()
	# addSaltAndPepperNoiseAndPrintImage()

	# LegendreTransformationTests
	# legendreTransformationDebug_Test()
	# legendreTransformationPrintPoints_Test()

	stop = timeit.default_timer()
	print('Time:', stop - start, "s")  

if __name__ == '__main__':
	main()