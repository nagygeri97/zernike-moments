import argparse
import numpy as np
import os
from enum import Enum
from PIL import Image, ImageDraw, ImageFont

from QZMI import *
from legendre.QZMILegendre import *
from ImageManipulation import *
from Utility import *

def templateMatchingTest():
	templatesFileName = "../images/lenna_pepper/lenna_color_512.bmp"
	matchingFileName = "../images/lenna_pepper/lenna_color_512.bmp"
	templateOutFile = "../tmpTemplate.bmp"
	templatePositions = [(123,431), (46,321), (256,121)] # TODO
	circleRadius = 11

	printImageWithTemplates(templatesFileName, circleRadius, templatePositions, templateOutFile)

def printImageWithTemplates(fileName, radius, positions, outFile = "../tmp.bmp"):
	img = Image.open(fileName)
	img.load()

	draw = ImageDraw.Draw(img)
	i = 1
	for (x,y) in positions:
		draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline='red')
		font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", int(1.5*radius))
		draw.text((x - int(radius/2), y - int(radius/1.3)), str(i), font=font, fill='red')
		i += 1
	img.save(outFile)

