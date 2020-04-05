import argparse
import numpy as np
import os
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict

from QZMI import *
from legendre.QZMILegendre import *
from ImageManipulation import *
from Utility import *

class TemplateType(Enum):
	PARK = 1
	FLAG = 2
	STREET = 3
	CARS = 4

def getTemplateData(templateType):
	path = "../images/templates/small/"
	outPath = "../results/images/template/"
	if templateType == TemplateType.PARK:
		templateFile = "park1.jpg"
		matchingFiles = ["park" + str(i) + ".jpg" for i in range(2,9)]
		templatePositions = [( 35,365), # Parking sign (missing in zoomed)
							 (207,442), # Bin
							 ( 79,434), # Red car rear right
							 (226,364), # Tree branches
							 (120,470), # Pole bottom
							 (147,263), # Building roof corner
							 (222,288), # Building window
							 (332,418), # Park entrance (missing in zoomed)
							 ( 93,388), # Building bottom
							]
	elif templateType == TemplateType.FLAG:
		templateFile = "flag1.jpg"
		matchingFiles = ["flag" + str(i) + ".jpg" for i in range(2,5)]
		templatePositions = [(400,220), # EU flag
							 (313,290), # Yellow car
							 (290, 35), # Building top open window (missing in zoomed)
							 ( 53,320), # Bin (missing in zoomed)
							 (225,204), # Tree top
							 (260,344), # Plants in grass
							 (130,270), # Sign
							 (264,205), # Balcony with lamppost
							 (332,308), # Bottom of post
							] 
	elif templateType == TemplateType.STREET:
		templateFile = "street1.jpg"
		matchingFiles = ["street" + str(i) + ".jpg" for i in range(2,6)]
		templatePositions = [( 75,450), # Bike rear
						     (328,394), # Bin
							 (193,434), # Dog bin
							 (210,378), # Fence
							 (290,364), # Toilet
							 (295,157), # Building top floor
							 (134,377), # Parked car
							 (134,407), # Hedge
							 (172,269), # Tree top
							]
	elif templateType == TemplateType.CARS:
		templateFile = "cars1.jpg"
		matchingFiles = ["cars" + str(i) + ".jpg" for i in range(2,4)]
		templatePositions = [(330,365), # White car light
							 (100,345), # Black car wheel
							 (144,152), # Balcony
							 (175,340), # Black car lamp
							 (360,555), # Manhole cover
							 (285,128), # Green building window
							 (135,285), # Yellow building door
							 (200,100), # Tree top
							 (384,375), # White car plate
							]
	return (path, templateFile, matchingFiles, templatePositions, outPath)

def templateMatchingTest():
	templateType = TemplateType.STREET
	qzmiClass = QZMI_NoCentroid

	(path, templateFile, matchingFiles, templatePositions, outPath) = getTemplateData(templateType)
	circleRadius = 10

	printImageWithTemplates(path + templateFile, circleRadius, templatePositions, outPath + templateFile[:-4] + "_template.bmp")
	# return
	
	img = getImgFromFileAsRawNpArray(path + templateFile)
	originalVecs = []
	for position in templatePositions:
		(x,y) = position
		croppedImg = np.array(img[(x - circleRadius):(x + circleRadius + 1), (y - circleRadius):(y + circleRadius + 1), :], dtype="double")
		originalVecs.append(populateInvariantVector(croppedImg, qzmiClass))
	
	for matchingFile in matchingFiles:
		# TODO: calculate scaling!
		matchingRadius = circleRadius
		img = getImgFromFileAsRawNpArray(path + matchingFile)
		(h, w, _) = img.shape
		matchingVecs = {}
		for x in range(matchingRadius, h - matchingRadius,5):
			print(x)
			for y in range(matchingRadius, w - matchingRadius,5):
				croppedImg = np.array(img[(x - matchingRadius):(x + matchingRadius + 1), (y - matchingRadius):(y + matchingRadius + 1), :], dtype="double")
				matchingVecs[(y,x)] = populateInvariantVector(croppedImg, qzmiClass)

		points = []
		for vec in originalVecs:
			minDist = -1
			minPos = (0,0)
			for pos, matchVec in matchingVecs.items():
				dist = vectorDistance(vec, matchVec)
				if minDist == -1 or dist < minDist:
					minDist = dist
					minPos = pos
			points.append(minPos)
		print(points)
		printImageWithTemplates(path + matchingFile, matchingRadius, points, outPath + matchingFile[:-4] + "_result.bmp")

		

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
