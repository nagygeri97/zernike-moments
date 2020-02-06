# Script for creating all transformed versions of an image.
import numpy as np
from PIL import Image

def placeImagesOnBackground():
	# change 96x72 img to 152x128 by adding a 28x28 black bar around
	names = ["36", "125", "127", "153", "157", "161", "259", "262", "308", "507", "514", "774", "875"]
	names = [name + ".png" for name in names]
	for name in names:
		img = Image.open("../images/cups/small/" + name)
		img.load()
		x1, y1, x2, y2 = 0-28, 0-28, 96+28, 72+28  # cropping coordinates
		bg = Image.new('RGB', (x2 - x1, y2 - y1), (0, 0, 0))
		bg.paste(img, (-x1, -y1))
		bg.save("../images/cups/extended/" + name)

def RST():
	"""
	Create the translated, rotated and scaled images
	"""
	# Parameters (change these)
	imageNames = ["36", "125", "127", "153", "157", "161", "259", "262", "308", "507", "514", "774", "875"]
	imageFormat = "png"
	imagePath = "../images/cups/extended/"
	resultPath = "../images/cups/transformed/"
	rotationStep = 30 # degrees, should divide 360
	xTranslation = 8 # pixels
	yTranslation = 5 # pixels
	minScale = 0.5
	maxScale = 2
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

def addGaussianNoise(img, mean, stddev):
	shape = img.shape
	newImg = np.round(img + np.random.normal(mean, stddev, shape))
	bounds = np.vectorize(lambda x : np.uint8((x if x > 0 else 0) if x < 255 else 255))
	return bounds(newImg)

def addSaltAndPepperNoise(img, density):
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

# if __name__ == '__main__':
# 	RST()
# 	placeImagesOnBackground()