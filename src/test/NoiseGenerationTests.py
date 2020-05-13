from PIL import Image

from ImageManipulation import *
from Utility import *

def addGaussianNoiseAndPrintImage():
	(img,_) = getImgFromFileAsNpArray('../images/coil/transformed/62x-11y9r270s1_25.png')
	newImg = addGaussianNoise(img,mean=0,stddev=7)
	im = Image.fromarray(newImg)
	im.save('../tdk/figures/noise/gauss7.png')

def addSaltAndPepperNoiseAndPrintImage():
	(img,_) = getImgFromFileAsNpArray('../images/coil/transformed/99x-11y9r240s0_75.png')
	newImg = addSaltAndPepperNoise(img,density=3)
	im = Image.fromarray(newImg)
	im.save('../tdk/figures/noise/pepper3.png')

def addGaussianNoiseFilterTest():
	(img,_) = getImgFromFileAsNpArray('../images/cups/extended/36.png')
	newImg = addGaussianNoise(img,mean=0,stddev=3)
	im = Image.fromarray(newImg)
	im.save('../before.png')
	newImg = meanFilter(newImg)
	im = Image.fromarray(newImg)
	im.save('../after.png')

def addSaltAndPepperNoiseFilterTest():
	(img,_) = getImgFromFileAsNpArray('../images/cups/extended/36.png')
	newImg = addSaltAndPepperNoise(img,density=5)
	im = Image.fromarray(newImg)
	im.save('../before.png')
	newImg = medianFilter(newImg)
	im = Image.fromarray(newImg)
	im.save('../after.png')