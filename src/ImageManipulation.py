import numpy as np
from PIL import Image, ImageFilter
import os

import Utility

# ------ Test image generation ---------

cupNames = ["36", "125", "127", "153", "157", "161", "259", "262", "308", "507", "514", "774", "875"]
coilNames = ["7", "13", "22", "26", "29", "32", "39", "55", "62", "64", "65", "71", "95", "99"]

def placeImagesOnBackground():
	# extend image by adding a black/grey bar around

	# Parameters (change these)
	barWitdh = 38 # pixels, 28 for cups, 38 for coil
	names = coilNames
	bgColor = (127,127,127) # (127,127,127) for grey, (0,0,0) for black
	inPath = "../images/coil/original/"
	outPath = "../images/coil/extended_grey/"
	# End changes here

	names = [name + ".png" for name in names]
	for name in names:
		img = Image.open(inPath + name)
		img.load()
		w,h = img.size
		x1, y1, x2, y2 = 0-barWitdh, 0-barWitdh, w+barWitdh, h+barWitdh  # cropping coordinates
		bg = Image.new('RGB', (x2 - x1, y2 - y1), bgColor)
		bg.paste(img, (-x1, -y1))
		bg.save(outPath + name)

def RST():
	"""
	Create the translated, rotated and scaled images
	"""
	# Parameters (change these)
	imageNames = coilNames
	imageFormat = "png"
	imagePath = "../images/coil/extended_grey/"
	resultPath = "../images/coil/transformed_grey/"
	rotationStep = 30 # degrees, should divide 360
	xTranslation = -11 # pixels
	yTranslation = 9 # pixels
	minScale = 0.5
	maxScale = 2.0
	scaleStep = 0.25
	bgColor = (127,127,127) # (127,127,127) for grey, (0,0,0) for black
	# End changes here

	angles = [angle for angle in range(0,360,rotationStep)] # in (degs,radians)
	scales = list(np.arange(minScale, maxScale, scaleStep))
	scales.append(maxScale)


	for imageName in imageNames:
		img = Image.open(imagePath + imageName + "." + imageFormat)
		img.load()

		# Translate
		imgAlpha = img.convert('RGBA')
		imgTtmp = imgAlpha.transform(img.size, Image.AFFINE, (1, 0, xTranslation, 0, 1, yTranslation))
		bgT = Image.new('RGBA', imgTtmp.size, (*bgColor, 255))
		imgT = Image.composite(imgTtmp, bgT, imgTtmp)
		imgT = imgT.convert(img.mode)
		images = []
		for angle in angles:
			imgAlpha = imgT.convert('RGBA')
			imgRtmp = imgAlpha.rotate(angle, resample = Image.BILINEAR)
			bgR = Image.new('RGBA', imgRtmp.size, (*bgColor, 255))
			imgR = Image.composite(imgRtmp, bgR, imgRtmp)
			imgR = imgR.convert(imgT.mode)

			for scale in scales:
				w, h = imgR.size
				imgRS = imgR.resize((int(w*scale), int(h*scale)), resample = Image.BILINEAR)
				bg = Image.new('RGB', imgRS.size, bgColor)
				bg.paste(imgRS, (0, 0))
				bg.save(resultPath + imageName + "x" + str(xTranslation) + "y" + str(yTranslation) + "r" + str(angle) + "s" + str(scale).replace('.','_') +  "." + imageFormat)

def rotate():
	"""
	Create the rotated images
	"""
	# Parameters (change these)
	imageNames = coilNames
	imageFormat = "png"
	imagePath = "../images/coil/extended_grey/"
	resultPath = "../images/coil/rotated_grey/"
	rotationStep = 5 # degrees, should divide 360
	bgColor = (127,127,127) # (127,127,127) for grey, (0,0,0) for black
	# End changes here

	angles = [angle for angle in range(0,360,rotationStep)] # in (degs,radians)


	for imageName in imageNames:
		img = Image.open(imagePath + imageName + "." + imageFormat)
		img.load()

		for angle in angles:
			imgAlpha = img.convert('RGBA')
			imgRtmp = imgAlpha.rotate(angle, resample = Image.BILINEAR)
			bgR = Image.new('RGBA', imgRtmp.size, (*bgColor, 255))
			imgR = Image.composite(imgRtmp, bgR, imgRtmp)
			imgR = imgR.convert(img.mode)
			imgR.save(resultPath + imageName + "r" + str(angle) + "." + imageFormat)


def squareImage(img, background = (0,0,0)):
	background = tuple(np.array(img)[0,0])
	w, h = img.size
	if w == h:
		return img
	elif w > h:
		diff = w - h
		offset1 = diff // 2
		offset2 = diff - offset1
		x1, y1, x2, y2 = 0, 0-offset1, w, h+offset2
		bg = Image.new('RGB', (x2 - x1, y2 - y1), background)
		bg.paste(img, (-x1, -y1))
		# bg.save("../tmp.bmp")
		return bg
	elif h > w:
		diff = h - w
		offset1 = diff // 2
		offset2 = diff - offset1
		x1, y1, x2, y2 = 0-offset1, 0, w+offset2, h
		bg = Image.new('RGB', (x2 - x1, y2 - y1), background)
		bg.paste(img, (-x1, -y1))
		# bg.save("../tmp.bmp")
		return bg

def scale():
	imageDir = "../images/templates/original/"
	outDir = "../images/templates/small/"
	scale = 1/5

	imageFiles = os.listdir(imageDir)
	
	for file in imageFiles:
		image = Image.open(imageDir + file)
		image.load()
		w, h = image.size
		image = image.resize((int(w*scale), int(h*scale)), resample = Image.BILINEAR).rotate(-90, expand = True)
		image.save(outDir + file)

# ------ Noise ---------

def addGaussianNoise(img, mean, stddev):
	# img: an np.array
	shape = img.shape
	newImg = np.round(img + np.random.normal(mean, stddev, shape))
	bounds = np.vectorize(lambda x : np.uint8((x if x > 0 else 0) if x < 255 else 255))
	# pilImg = Image.fromarray(bounds(newImg))
	# pilImg.save("../test.bmp", "BMP")
	return bounds(newImg)

def addSaltAndPepperNoise(img, density):
	# img: an np.array
	# density is the PERCENTAGE of pixels affected
	density = float(density) / 100
	(row, col, ch) = img.shape
	amount = round(row*col*density)
	addSalt = True
	for i in range(amount):
		randRow = np.random.randint(0,row)
		randCol = np.random.randint(0,col)
		if addSalt:
			img[randRow,randCol] = [255,255,255]
		else:
			img[randRow,randCol] = [0,0,0]
		addSalt = not addSalt
	return img

def addGaussianNoiseNoRounding(img, mean, stddev):
	shape = img.shape
	newImg = np.zeros(shape, dtype='double')
	newImg += img + np.random.normal(mean, stddev, shape)
	return newImg

def medianFilter(img):
	# img: an np.array
	pilImg = Image.fromarray(img)
	pilImg = pilImg.filter(ImageFilter.MedianFilter())
	img = np.array(pilImg)
	return img

def gaussianBlur(img):
	# img: an np.array
	pilImg = Image.fromarray(img)
	pilImg = pilImg.filter(ImageFilter.GaussianBlur())
	img = np.array(pilImg)
	return img

def meanFilter(img):
	# img: an np.array
	pilImg = Image.fromarray(img)
	pilImg = pilImg.filter(ImageFilter.Kernel((3,3), np.ones(9)))
	img = np.array(pilImg)
	return img

def addGaussianNoiseFiltered(img, mean, stddev):
	img = addGaussianNoise(img, mean, stddev)
	img = meanFilter(img)
	return img

def addSaltAndPepperNoiseFiltered(img, density):
	img = addSaltAndPepperNoise(img, density)
	img = medianFilter(img)
	return img

# ------ Centroid ---------

def centroidTranslation(img):
	# valoszinuleg rossz?
	# img: an np.array
	pilImg = Image.fromarray(img)

	(cy, cx) = Utility.calculateCentroid(img)
	N, M = pilImg.size

	xTranslation = cx - (N//2)
	yTranslation = cy - (M//2) # N, M may be mixed up

	pilImg = pilImg.transform(pilImg.size, Image.AFFINE, (1, 0, xTranslation, 0, 1, yTranslation))

	# pilImg.save("../test.bmp", "BMP")
	img = np.array(pilImg)
	return img

def centroidTranslationFloat(img):
	# ez talan jo
	(cx, cy) = Utility.calculateCentroid(img)
	N, M, _ = img.shape
	xTranslation = (cx - (N//2)) # kell a minusz ele?
	yTranslation = (cy - (M//2))

	newImg = np.zeros((N,M,3),dtype='double')
	for x in range(N):
		for y in range(M):
			if x - xTranslation > 0 and y - yTranslation > 0 and x - xTranslation < N and y - yTranslation < M:
				for i in range(3):
					newImg[x - xTranslation, y - yTranslation, i] = img[x,y,i]
	
	# pilImg = Image.fromarray(newImg)
	# pilImg.save("../test2.bmp", "BMP")
	return newImg

# ------ RGB ---------

def getColorComponent(img, color='R'):
	if color == 'R':
		index = 0
	elif color == 'G':
		index = 1
	elif color == 'B':
		index = 2
	else:
		return
	(heigth, width, _) = img.shape
	monochromeImg = np.empty((heigth, width), dtype='double')
	for i in range(heigth):
		for j in range(width):
			monochromeImg[i,j] = img[i,j][index]

	return monochromeImg

# ------ Conversion --------

def imageToFloat(img):
	return img.astype('double')

if __name__ == '__main__':
	# placeImagesOnBackground()
	# RST()
	# rotate()
	scale()