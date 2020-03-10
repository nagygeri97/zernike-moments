import numpy as np
from PIL import Image, ImageFilter

import Utility

# ------ Test image generation ---------

cupNames = ["36", "125", "127", "153", "157", "161", "259", "262", "308", "507", "514", "774", "875"]
coilNames = ["7", "13", "22", "26", "29", "32", "39", "55", "62", "64", "65", "71", "95", "99"]

def placeImagesOnBackground():
	# extend image by adding a black bar around
	barWitdh = 38 # 28 for cups, 38 for coil
	names = coilNames # Change this to the correct name
	names = [name + ".png" for name in names]
	for name in names:
		img = Image.open("../images/coil/original/" + name) # Change to the correct path
		img.load()
		w,h = img.size
		x1, y1, x2, y2 = 0-barWitdh, 0-barWitdh, w+barWitdh, h+barWitdh  # cropping coordinates
		bg = Image.new('RGB', (x2 - x1, y2 - y1), (0, 0, 0))
		bg.paste(img, (-x1, -y1))
		bg.save("../images/coil/extended/" + name) # Change to the correct path

def RST():
	"""
	Create the translated, rotated and scaled images
	"""
	# Parameters (change these)
	imageNames = coilNames
	imageFormat = "png"
	imagePath = "../images/coil/extended/"
	resultPath = "../images/coil/transformed/"
	rotationStep = 30 # degrees, should divide 360
	xTranslation = -11 # pixels
	yTranslation = 9 # pixels
	minScale = 0.5
	maxScale = 2.0
	scaleStep = 0.25
	# End changes here

	angles = [angle for angle in range(0,360,rotationStep)] # in (degs,radians)
	scales = list(np.arange(minScale, maxScale, scaleStep))
	scales.append(maxScale)


	for imageName in imageNames:
		img = Image.open(imagePath + imageName + "." + imageFormat)
		img.load()

		# Translate
		img = img.transform(img.size, Image.AFFINE, (1, 0, xTranslation, 0, 1, yTranslation))
		images = []
		for angle in angles:
			imgR = img.rotate(angle, resample = Image.BILINEAR)
			for scale in scales:
				w, h = imgR.size
				imgRS = imgR.resize((int(w*scale), int(h*scale)), resample = Image.BILINEAR)
				bg = Image.new('RGB', imgRS.size, (0, 0, 0))
				bg.paste(imgRS, (0, 0))
				bg.save(resultPath + imageName + "x" + str(xTranslation) + "y" + str(yTranslation) + "r" + str(angle) + "s" + str(scale).replace('.','_') +  "." + imageFormat)

def squareImage(img, background = (0,0,0)):
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
		return bg
	elif h > w:
		diff = h - w
		offset1 = diff // 2
		offset2 = diff - offset1
		x1, y1, x2, y2 = 0-offset1, 0, w+offset2, h
		bg = Image.new('RGB', (x2 - x1, y2 - y1), background)
		bg.paste(img, (-x1, -y1))
		return bg

# ------ Noise ---------

def addGaussianNoise(img, mean, stddev):
	# img: an np.array
	shape = img.shape
	newImg = np.round(img + np.random.normal(mean, stddev, shape))
	bounds = np.vectorize(lambda x : np.uint8((x if x > 0 else 0) if x < 255 else 255))
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

# if __name__ == '__main__':
	# RST()
	# placeImagesOnBackground()