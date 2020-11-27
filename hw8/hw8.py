import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import math

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = len(img) # row
width = len(img[0]) # col, it's actually a square though
kernel = np.ones((5, 5), dtype = np.uint8)

def initializeKernel(kernel):
	kernel[0, 0] = 0
	kernel[0, 4] = 0
	kernel[4, 0] = 0
	kernel[4, 4] = 0
	return kernel

def gaussianNoise(image, amplitude): # I(nim, i, j) = I(im, i, j) + amplitude*N(0,1) , amplitude will be 10 and 30
	ret = np.zeros(image.shape, np.int)
	for i in range(int(width)):
		for j in range(int(length)):
			ret[i, j] = image[i, j] + random.gauss(0, 1) * amplitude
	
	return ret
	
def saltAndPepperNoise(image, ratio): # ratio is threshold, and then it will be 0.05 and 0.1 in this hw
	ret = np.zeros(image.shape, np.int)
	for i in range(int(width)):
		for j in range(int(length)):
			rolling_dice = random.uniform(0, 1)
			if rolling_dice < ratio:
				ret[i, j] = 0 # pepper
			elif rolling_dice > 1 - ratio:
				ret[i, j] = 255
			else:
				ret[i, j] = image[i, j]
	return ret
	
def boxFilter(image, size): # size = 3 if mask is 3*3, size = 5 if 5*5
	center = int(size // 2)
	ret = np.zeros(image.shape, np.int)
	mask = np.zeros((size, size), np.float) # debug: remember!! np.float!!!
	for i in range(size): # adjust mask
		for j in range(size):
			mask[i, j] = 1 / (size * size)

	for i in range(int(width)):
		for j in range(int(length)):
			temp = 0
			for kernel_i in range(size):
				for kernel_j in range(size):
					scope_i = i + kernel_i - center
					scope_j = j + kernel_j - center
					if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
						temp += image[scope_i, scope_j] * mask[kernel_i, kernel_j]
			ret[i, j] = int(temp)
	return ret
	
def medianFilter(image, size): # size = 3 if mask is 3*3, size = 5 if 5*5
	center = int(size // 2)
	ret = np.zeros(image.shape, np.int)

	for i in range(int(width)):
		for j in range(int(length)):
			temp = []
			for kernel_i in range(size):
				for kernel_j in range(size):
					scope_i = i + kernel_i - center
					scope_j = j + kernel_j - center
					if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
						temp.append(image[scope_i, scope_j])
			temp.sort()
			if len(temp) & 0 == 1: # represent even number
				ret[i, j] = temp[int((len(temp) - 1) // 2)] + temp[int((len(temp) - 1) // 2) + 1]
			else: # odd
				ret[i, j] = temp[int((len(temp) - 1) // 2)]
	return ret	
	
def openThenClose(image, kernel): # erosion, dilation, dilation, erosion
	eroted_photo = erosion(image, kernel)
	open = dilation(eroted_photo, kernel)
	dilated_photo = dilation(open, kernel)
	close = erosion(dilated_photo, kernel)
	return close
	
def closeThenOpen(image, kernel): # dilation, erosion, erosion, dilation
	dilated_photo = dilation(image, kernel)
	close = erosion(dilated_photo, kernel)
	eroted_photo = erosion(close, kernel)
	open = dilation(eroted_photo, kernel)	
	return open
	
def SNR(clear_image, noised_image):
	average_pixel_value = 0
	average_noise_pixel_value = 0
	temp = 0
	for i in range(int(width)):
		for j in range(int(length)):
			average_pixel_value += clear_image[i, j]
			temp = int(noised_image[i, j]) - int(clear_image[i, j])
			average_noise_pixel_value += temp
			
	average_pixel_value /= (width * length)
	average_noise_pixel_value /= (width * length)
	
	VS = 0
	VN = 0
	for i in range(int(width)):
		for j in range(int(length)):
			VS += (int(clear_image[i, j]) - int(average_pixel_value)) ** 2
			VN += (int(noised_image[i, j]) - int(clear_image[i, j]) - int(average_noise_pixel_value)) ** 2 # int must be added! or will be error
	VS /= (width * length)
	VN /= (width * length)
	
	return 20 * math.log10(math.sqrt(VS)/math.sqrt(VN))
	
def dilation(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			_max = []
			for kernel_i in range(5):
				for kernel_j in range(5):
					scope_i = i + kernel_i - 2
					scope_j = j + kernel_j - 2
					if kernel[kernel_i, kernel_j] == 1: # if some position is 1 in structure component, then do the following
						if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
							_max.append(image[scope_i, scope_j])
			ret[i, j] = max(_max)
	return ret

def erosion(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			_min = []
			for kernel_i in range(5):
				for kernel_j in range(5):
					scope_i = i + kernel_i - 2
					scope_j = j + kernel_j - 2
					if kernel[kernel_i, kernel_j] == 1: # if some position is 1 in structure component, then do the following
						if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
							_min.append(image[scope_i, scope_j])
			ret[i, j] = min(_min)
	return ret
	
# set kernel	
kernel = initializeKernel(kernel)
# save SNR value into SNR.txt
file = open("SNR.txt", "w")

# gaussian noise 10 and 30 (2 images)
gaussianNoise_10_img = gaussianNoise(img, 10)
cv2.imwrite('gaussianNoise_10_img.png', gaussianNoise_10_img)
file.write("gaussianNoise_10_img 0.05 SNR:" + str(SNR(img, gaussianNoise_10_img)) + '\n')

gaussianNoise_30_img = gaussianNoise(img, 30)
cv2.imwrite('gaussianNoise_30_img.png', gaussianNoise_30_img)
file.write("gaussianNoise_30_img 0.05 SNR:" + str(SNR(img, gaussianNoise_30_img)) + '\n')

# salt and pepper noise 0.05 and 0.1 (2 images)
saltAndPepperNoise_005_img = saltAndPepperNoise(img, 0.05)
cv2.imwrite('saltAndPepperNoise_005_img.png', saltAndPepperNoise_005_img)
file.write("saltAndPepperNoise_0.05_img 0.05 SNR:" + str(SNR(img, saltAndPepperNoise_005_img)) + '\n')

saltAndPepperNoise_01_img = saltAndPepperNoise(img, 0.1)
cv2.imwrite('saltAndPepperNoise_01_img.png', saltAndPepperNoise_01_img)
file.write("saltAndPepperNoise_0.1_img 0.05 SNR:" + str(SNR(img, saltAndPepperNoise_01_img)) + '\n')

# Use the 3x3, 5x5 box filter on images (8 images)
boxFilter_3_gaussianNoise_10_img = boxFilter(gaussianNoise_10_img, 3)
cv2.imwrite('boxFilter_3_gaussianNoise_10_img.png', boxFilter_3_gaussianNoise_10_img)
file.write("boxFilter_3_gaussianNoise_10_img SNR:" + str(SNR(img, boxFilter_3_gaussianNoise_10_img)) + '\n')

boxFilter_3_gaussianNoise_30_img = boxFilter(gaussianNoise_30_img, 3)
cv2.imwrite('boxFilter_3_gaussianNoise_30_img.png', boxFilter_3_gaussianNoise_30_img)
file.write("boxFilter_3_gaussianNoise_30_img SNR:" + str(SNR(img, boxFilter_3_gaussianNoise_30_img)) + '\n')

boxFilter_3_saltAndPepperNoise_005_img = boxFilter(saltAndPepperNoise_005_img, 3)
cv2.imwrite('boxFilter_3_saltAndPepperNoise_005_img.png', boxFilter_3_saltAndPepperNoise_005_img)
file.write("boxFilter_3_saltAndPepperNoise_005_img SNR:" + str(SNR(img, boxFilter_3_saltAndPepperNoise_005_img)) + '\n')

boxFilter_3_saltAndPepperNoise_01_img = boxFilter(saltAndPepperNoise_01_img, 3)
cv2.imwrite('boxFilter_3_saltAndPepperNoise_01_img.png', boxFilter_3_saltAndPepperNoise_01_img)
file.write("boxFilter_3_saltAndPepperNoise_01_img SNR:" + str(SNR(img, boxFilter_3_saltAndPepperNoise_01_img)) + '\n')

boxFilter_5_gaussianNoise_10_img = boxFilter(gaussianNoise_10_img, 5)
cv2.imwrite('boxFilter_5_gaussianNoise_10_img.png', boxFilter_5_gaussianNoise_10_img)
file.write("boxFilter_5_gaussianNoise_10_img SNR:" + str(SNR(img, boxFilter_5_gaussianNoise_10_img)) + '\n')

boxFilter_5_gaussianNoise_30_img = boxFilter(gaussianNoise_30_img, 5)
cv2.imwrite('boxFilter_5_gaussianNoise_30_img.png', boxFilter_5_gaussianNoise_30_img)
file.write("boxFilter_5_gaussianNoise_30_img SNR:" + str(SNR(img, boxFilter_5_gaussianNoise_30_img)) + '\n')

boxFilter_5_saltAndPepperNoise_005_img = boxFilter(saltAndPepperNoise_005_img, 5)
cv2.imwrite('boxFilter_5_saltAndPepperNoise_005_img.png', boxFilter_5_saltAndPepperNoise_005_img)
file.write("boxFilter_5_saltAndPepperNoise_005_img SNR:" + str(SNR(img, boxFilter_5_saltAndPepperNoise_005_img)) + '\n')

boxFilter_5_saltAndPepperNoise_01_img = boxFilter(saltAndPepperNoise_01_img, 5)
cv2.imwrite('boxFilter_5_saltAndPepperNoise_01_img.png', boxFilter_5_saltAndPepperNoise_01_img)
file.write("boxFilter_5_saltAndPepperNoise_01_img SNR:" + str(SNR(img, boxFilter_5_saltAndPepperNoise_01_img)) + '\n')

# Use 3x3, 5x5 median filter on images(8 images)
medianFilter_3_gaussianNoise_10_img = medianFilter(gaussianNoise_10_img, 3)
cv2.imwrite('medianFilter_3_gaussianNoise_10_img.png', medianFilter_3_gaussianNoise_10_img)
file.write("medianFilter_3_gaussianNoise_10_img SNR:" + str(SNR(img, medianFilter_3_gaussianNoise_10_img)) + '\n')

medianFilter_3_gaussianNoise_30_img = medianFilter(gaussianNoise_30_img, 3)
cv2.imwrite('medianFilter_3_gaussianNoise_30_img.png', medianFilter_3_gaussianNoise_30_img)
file.write("medianFilter_3_gaussianNoise_30_img SNR:" + str(SNR(img, medianFilter_3_gaussianNoise_30_img)) + '\n')

medianFilter_3_saltAndPepperNoise_005_img = medianFilter(saltAndPepperNoise_005_img, 3)
cv2.imwrite('medianFilter_3_saltAndPepperNoise_005_img.png', medianFilter_3_saltAndPepperNoise_005_img)
file.write("medianFilter_3_saltAndPepperNoise_005_img SNR:" + str(SNR(img, medianFilter_3_saltAndPepperNoise_005_img)) + '\n')

medianFilter_3_saltAndPepperNoise_01_img = medianFilter(saltAndPepperNoise_01_img, 3)
cv2.imwrite('medianFilter_3_saltAndPepperNoise_01_img.png', medianFilter_3_saltAndPepperNoise_01_img)
file.write("medianFilter_3_saltAndPepperNoise_01_img SNR:" + str(SNR(img, medianFilter_3_saltAndPepperNoise_01_img)) + '\n')

medianFilter_5_gaussianNoise_10_img = medianFilter(gaussianNoise_10_img, 5)
cv2.imwrite('medianFilter_5_gaussianNoise_10_img.png', medianFilter_5_gaussianNoise_10_img)
file.write("medianFilter_5_gaussianNoise_10_img SNR:" + str(SNR(img, medianFilter_5_gaussianNoise_10_img)) + '\n')

medianFilter_5_gaussianNoise_30_img = medianFilter(gaussianNoise_30_img, 5)
cv2.imwrite('medianFilter_5_gaussianNoise_30_img.png', medianFilter_5_gaussianNoise_30_img)
file.write("medianFilter_5_gaussianNoise_30_img SNR:" + str(SNR(img, medianFilter_5_gaussianNoise_30_img)) + '\n')

medianFilter_5_saltAndPepperNoise_005_img = medianFilter(saltAndPepperNoise_005_img, 5)
cv2.imwrite('medianFilter_5_saltAndPepperNoise_005_img.png', medianFilter_5_saltAndPepperNoise_005_img)
file.write("medianFilter_5_saltAndPepperNoise_005_img SNR:" + str(SNR(img, medianFilter_5_saltAndPepperNoise_005_img)) + '\n')

medianFilter_5_saltAndPepperNoise_01_img = medianFilter(saltAndPepperNoise_01_img, 5)
cv2.imwrite('medianFilter_5_saltAndPepperNoise_01_img.png', medianFilter_5_saltAndPepperNoise_01_img)
file.write("medianFilter_5_saltAndPepperNoise_01_img SNR:" + str(SNR(img, medianFilter_5_saltAndPepperNoise_01_img)) + '\n')

# Use both opening-then-closing and closing-then opening filter (4 + 4 images)
openThenClose_gaussianNoise_10_img = openThenClose(gaussianNoise_10_img, kernel)
cv2.imwrite('openThenClose_gaussianNoise_10_img.png', openThenClose_gaussianNoise_10_img)
file.write("openThenClose_gaussianNoise_10_img SNR:" + str(SNR(img, openThenClose_gaussianNoise_10_img)) + '\n')

openThenClose_gaussianNoise_30_img = openThenClose(gaussianNoise_30_img, kernel)
cv2.imwrite('openThenClose_gaussianNoise_30_img.png', openThenClose_gaussianNoise_30_img)
file.write("openThenClose_gaussianNoise_30_img SNR:" + str(SNR(img, openThenClose_gaussianNoise_30_img)) + '\n')

openThenClose_saltAndPepperNoise_005_img = openThenClose(saltAndPepperNoise_005_img, kernel)
cv2.imwrite('openThenClose_saltAndPepperNoise_005_img.png', openThenClose_saltAndPepperNoise_005_img)
file.write("openThenClose_saltAndPepperNoise_005_img SNR:" + str(SNR(img, openThenClose_saltAndPepperNoise_005_img)) + '\n')

openThenClose_saltAndPepperNoise_01_img = openThenClose(saltAndPepperNoise_01_img, kernel)
cv2.imwrite('openThenClose_saltAndPepperNoise_01_img.png', openThenClose_saltAndPepperNoise_01_img)
file.write("openThenClose_saltAndPepperNoise_01_img SNR:" + str(SNR(img, openThenClose_saltAndPepperNoise_01_img)) + '\n')

closeThenOpen_gaussianNoise_10_img = closeThenOpen(gaussianNoise_10_img, kernel)
cv2.imwrite('closeThenOpen_gaussianNoise_10_img.png', closeThenOpen_gaussianNoise_10_img)
file.write("closeThenOpen_gaussianNoise_10_img SNR:" + str(SNR(img, closeThenOpen_gaussianNoise_10_img)) + '\n')

closeThenOpen_gaussianNoise_30_img = closeThenOpen(gaussianNoise_30_img, kernel)
cv2.imwrite('closeThenOpen_gaussianNoise_30_img.png', closeThenOpen_gaussianNoise_30_img)
file.write("closeThenOpen_gaussianNoise_30_img SNR:" + str(SNR(img, closeThenOpen_gaussianNoise_30_img)) + '\n')

closeThenOpen_saltAndPepperNoise_005_img = closeThenOpen(saltAndPepperNoise_005_img, kernel)
cv2.imwrite('closeThenOpen_saltAndPepperNoise_005_img.png', closeThenOpen_saltAndPepperNoise_005_img)
file.write("closeThenOpen_saltAndPepperNoise_005_img SNR:" + str(SNR(img, closeThenOpen_saltAndPepperNoise_005_img)) + '\n')

closeThenOpen_saltAndPepperNoise_01_img = closeThenOpen(saltAndPepperNoise_01_img, kernel)
cv2.imwrite('closeThenOpen_saltAndPepperNoise_01_img.png', closeThenOpen_saltAndPepperNoise_01_img)
file.write("closeThenOpen_saltAndPepperNoise_01_img SNR:" + str(SNR(img, closeThenOpen_saltAndPepperNoise_01_img)) + '\n')

file.close()