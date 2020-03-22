import numpy as np

from Utility import *
from legendre.TransformationsLegendre import *

def legendreTransformationPrintPoints_Test():
	img, _ = getImgFromFileAsNpArray("../images/coil/extended/55.png")
	lt = LegendreTransformation1(img)
	printImageFromLegendreTrans(lt, "../test.bmp")
	# import pdb; pdb.set_trace()

def legendreTransformationDebug_Test():
	img, _ = getImgFromFileAsNpArray("../images/coil/extended/55.png")
	lt = LegendreTransformation1(img)
	# import pdb; pdb.set_trace()

def legendreTransformation2PrintPoints_Test():
	img, _ = getImgFromFileAsNpArray("../images/coil/extended/55.png")
	lt = LegendreTransformation2(img)
	printImageFromLegendreTrans(lt, "../test.bmp", 100)
	# import pdb; pdb.set_trace()