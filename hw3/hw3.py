import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though
	
def histogramPixelCount(image):
	ret = [0 for _ in range(256)]
	for i in range(int(length)):
		for j in range(int(width)):
			ret[image[i][j]] += 1 # pixel 0 ~ 255 will be put in corresponding index in "ret" list 
	return ret

def pixelsDivideBy3(image):
	ret = np.zeros(image.shape, dtype = np.int32)
	for i in range(int(length)):
		for j in range(int(width)):
			ret[i][j] = round(image[i][j] / 3)
	return ret
	
def makehistogramEqualizationTable(pixels):
	ret = [0 for _ in range(256)]
	for i in range(len(pixels)):
		for j in range(0, i + 1):
			ret[i] += pixels[j]
		ret[i] = round(255 * (ret[i] / (length * width)))
	return ret

def histogramEqualize(image, pixels):
	ret = np.zeros(image.shape, np.int)
	for i in range(int(length)):
		for j in range(int(width)):	
			ret[i][j] = pixels[image[i][j]]
	return ret
	

# original histogram picture output
pixel_count1 = histogramPixelCount(img)
plt.bar(range(256), pixel_count1, width = 1, facecolor = "blue", edgecolor = 'white')
plt.savefig("histogram_original.png")
plt.close() # remember add it! or photo will overlap
cv2.imwrite('original.png', img)

# picture every pixel divide by 3
divide_by_3_picture = pixelsDivideBy3(img)
pixel_count2 = histogramPixelCount(divide_by_3_picture)
plt.bar(range(256), pixel_count2, width = 1, facecolor = "red", edgecolor = 'white')
plt.savefig("histogram_divide_by_3.png")
plt.close()
cv2.imwrite('divide_by_3.png', divide_by_3_picture)

# after equalization
table = makehistogramEqualizationTable(pixel_count2)
equalized_picture = histogramEqualize(divide_by_3_picture, table)
pixel_count3 = histogramPixelCount(equalized_picture)
plt.bar(range(256), pixel_count3, width = 1, facecolor = "green", edgecolor = 'white')
plt.savefig("histogram_equalized_picture.png")
plt.close()
cv2.imwrite('equalized_picture.png', equalized_picture)