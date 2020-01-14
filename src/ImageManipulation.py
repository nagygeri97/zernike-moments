# Script for creating all transformed versions of an image.
import numpy as np
from PIL import Image

# Parameters (change these)
imagePath = "../images/cups/original/small/36.png"
resultPath = "../images/cups/transformed/"
rotationStep = 30 # degrees, should divide 360
xTranslation = 8 # pixels
yTranslation = 5 # pixels
minScale = 0.5
maxScale = 2
scaleStep = 0.25
# End changes here

angles = [angle * 2 * np.pi / 360.0 for angle in range(0,360,rotationStep)] # in radians
scales = list(np.arange(minScale, maxScale, scaleStep))
scales.append(maxScale)


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

def manipulate():
	img = Image.open(imagePath)
	img.load()

	# Translate
	img = img.transform(img.size, Image.AFFINE, (1, 0, xTranslation, 0, 1, yTranslation))
	img.save(resultPath + "test.png")


if __name__ == '__main__':
	# manipulate()
	placeImagesOnBackground()