import numpy as np

from Utility import *
from legendre.TransformationsLegendre import *

def legendreTransformationPrintPoints_Test():
	# img, _ = getImgFromFileAsNpArray("../images/coil/extended/55.png")
	img, _ = getImgFromFileAsNpArray("../images/lenna_pepper/pepper_color_64.bmp")
	img = np.array(img, dtype="double")
	lt = LegendreTransformation1(img)
	printImageFromLegendreTrans(lt, "../test.png",256)
	# import pdb; pdb.set_trace()

def legendreTransformationDebug_Test():
	img, _ = getImgFromFileAsNpArray("../images/coil/extended/55.png")
	lt = LegendreTransformation1(img)
	# import pdb; pdb.set_trace()

def legendreTransformation2PrintPoints_Test():
	# img, _ = getImgFromFileAsNpArray("../images/coil/extended/55.png")
	img, _ = getImgFromFileAsNpArray("../images/lenna_pepper/pepper_color_64.bmp")
	img = np.array(img, dtype="double")
	lt = LegendreTransformation2(img)
	printImageFromLegendreTrans(lt, "../test.png", 128)
	# import pdb; pdb.set_trace()