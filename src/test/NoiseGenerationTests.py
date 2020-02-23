from PIL import Image

from ImageManipulation import *
from Utility import *

def addGaussianNoiseAndPrintImage():
	(img,_) = getImgFromFileAsNpArray('../images/cups/extended/36.png')
	newImg = addGaussianNoise(img,mean=0,stddev=3)
	im = Image.fromarray(newImg)
	im.save('../test.bmp', "BMP")

def addSaltAndPepperNoiseAndPrintImage():
	(img,_) = getImgFromFileAsNpArray('../images/cups/extended/36.png')
	newImg = addSaltAndPepperNoise(img,density=5)
	im = Image.fromarray(newImg)
	im.save('../test.bmp', "BMP")