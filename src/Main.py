#!/usr/bin/env python3
# import timeit

from Utility import *

from test.InvarianceTests import *
from test.LegendreTransformationTests import *
from test.NoiseGenerationTests import *
from test.RecognitionTests import *
from test.ReconstructionTests import *
from test.TemplateMatchingTests import *

def main():
	# start = timeit.default_timer()

	# ReconstructionTests
	# testImageReconstructionTf1()
	# testImageReconstructionTf2()
	# testImageReconstructionLegendre1()
	# testImageReconstructionLegendre2()
	# testImageReconstructionErrors()
	
	# InvarianceTests
	# testInvariance()
	# testLegendreInvariance()

	# RecognitionTests
	# runRecognitionTest()

	# NoiseGenerationTests
	# addGaussianNoiseAndPrintImage()
	# addSaltAndPepperNoiseAndPrintImage()
	# addGaussianNoiseFilteTest()
	# addSaltAndPepperNoiseFilterTest()

	# LegendreTransformationTests
	# legendreTransformationDebug_Test()
	# legendreTransformationPrintPoints_Test()
	# legendreTransformation2PrintPoints_Test()
	# printCircleGrid()

	# TemplateMatchingTests
	templateMatchingTest()

	# stop = timeit.default_timer()
	# print('Time:', stop - start, "s")  

if __name__ == '__main__':
	main()