import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

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
	
def binarize(image):
	ret = np.zeros(image.shape, np.int)
	for i in range(int(width)):
		for j in range(int(length)):
			if image[j][i] >= 128: # 128 ~ 255
				ret[j][i] = 255
			else: # 0 ~ 127
				ret[j][i] = 0
	return ret

def dilation(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			value = 0
			for kernel_i in range(5):
				for kernel_j in range(5):
					scope_i = i + kernel_i - 2
					scope_j = j + kernel_j - 2
					if kernel[kernel_i, kernel_j] == 1: # if some position is 1 in structure component, then do the following
						if (0 <= scope_i and scope_i < length) and (0 <= scope_j and scope_j < width):
							if image[scope_i][scope_j] == 255: # if the position kernel == 1 and one of the 
								value = 255 # make it white
			ret[i, j] = value
	return ret
					
''' test another way for dilation
def try_d(image, kernel):
	ret = np.zeros(image.shape, np.int)
	# center of kernel will be put from [[2, width - 1 - 2], [2, length - 1 - 2]] becuz kernel center is in (2, 2)
	# hence, I can imagine kernel (0, 0) will be put in the (-2, -2) of the original photo (Lena) 
	for i in range(int(width)):
		for j in range(int(length)):
			if image[i, j] == 255:
				for kernel_i in range(-2, 3, 1):
					for kernel_j in range(-2, 3, 1):
						if (0 <= i + kernel_i and i + kernel_i < length) and (0 <= j + kernel_j and j + kernel_j < width):	
							ret[i + kernel_i, j + kernel_j] = 255
	return ret				
'''

# set kernel	
kernel = initializeKernel(kernel)
# get binary photo
binarized_photo = binarize(img)
# dilation
dilated_photo = dilation(binarized_photo, kernel)
cv2.imwrite('dilation.png', dilated_photo)