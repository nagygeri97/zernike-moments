# Script for creating all transformed versions of an image.
import numpy as np
from PIL import Image

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


def placeImagesOnBackground():
	# change 96x72 img to 152x128 by adding a 28x28 black bar around
	names = imageNames
	names = [name + ".png" for name in names]
	for name in names:
		img = Image.open("../images/cups/small/" + name)
		img.load()
		x1, y1, x2, y2 = 0-28, 0-28, 96+28, 72+28  # cropping coordinates
		bg = Image.new('RGB', (x2 - x1, y2 - y1), (0, 0, 0))
		bg.paste(img, (-x1, -y1))
		bg.save("../images/cups/extended/" + name)

def manipulate():
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


if __name__ == '__main__':
	manipulate()
	# placeImagesOnBackground()